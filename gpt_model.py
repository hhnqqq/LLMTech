import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import time
import json
import traceback

import torch
import pandas as pd
import numpy as np
import sys
import random

import utils

from typing import Optional, Union

from pydantic import BaseModel
from tqdm import tqdm
from torch import nn,optim



device = torch.device("cuda")
# device = torch.device("cpu")
dict_datas = json.load(open('./dict_datas.json', 'r'))
word2id, id2word = dict_datas['word2id'], dict_datas['id2word']
class ParaModel(BaseModel):
    max_pos: int = 1800 # 最大长度为1800
    hidden_size: int  = 768 # decoder layer的隐藏层大小为768
    attention_heads: int = 8 # 8头注意力机制
    attention_dim: int = 64 # 注意力维度为64
    fnn_dim: int = 2048 # fnn的中间层大小为2048
    n_layers: int = 6 # 6个decoder块
    clip:int = 1
    vocab_size: int
    max_new_tokens: Optional[int]
    dtype: Optional[str]
    max_tokens: Optional[str]
    
model_parameters = ParaModel(vocab_size=len(word2id))

def get_atten_pad_mask(seq_q: torch.Tensor) -> torch.Tensor:  
    '''
    作用：给定一个填充后的张量，返回对应的填充掩码。
    输入：一个形状为 [batch_size, len_q] 的张量。
    输出：一个形状为 [batch_size, len_q, len_q] 的 0/1 张量，其中填充位置为 1，其余位置为 0。
    '''
    batch_size, len_q = seq_q.size()
    pad_atten_mask = seq_q.data.eq(0).unsqueeze(1)
    return pad_atten_mask.expand(batch_size,len_q,len_q)


def get_attn_subsequence_mask(seq):
    '''
    作用：生成一个上三角的mask矩阵
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask=subsequence_mask.to(device)
    return subsequence_mask # [batch_size, tgt_len, tgt_len]


class GetScaledAttenScore(nn.Module):
    # 这个类的作用是得到Q,K,V结果之后计算注意力分数以及注意力的最终输出
    def __init__(self):
        # 执行nn.module的初始化方法
        super(GetScaledAttenScore,self).__init__()
    
    def forward(self, Q, K, V, weight_o, atten_musks, batch_size):
        """
        input的维度为[input_nums,embedding_size]
        wq,wk,wv的维度为[embedding_size ,attention_dim]
        q,k,v的维度为[batch_size,attention_heads, input_nums, attention_dim]
        attention分数的维度为[batch_size,attention_heads,input_nums,input_nums]
        attention结果的维度为[batch_size,attention_heads,input_nunms,attention_dim]
        """
        # 点积的大小和注意力矩阵的维度相关，为了保持数值稳定性，要除根号下的注意力维度
        atten_scores = torch.matmul(Q,K.transpose(-1,-2)) / np.sqrt(model_parameters.attention_dim) # 注意力分数，维度为[batch_size,n_heads,input_nums,input_nums] 和mask的维度刚好相同
        # Decoder-only结构的后向musk
        atten_scores.masked_fill_(atten_musks, -1e9) # tensor.masked_fill_(mask, value)，将标记为true的位置替换为-1e9
        
        # 对attention分数进行softmax操作
        atten_scores = nn.Softmax(dim=-1)(atten_scores) # 新增一个softmax层并且乘上去，没有参数，所以不需要初始化
        atten_result = torch.matmul(atten_scores,V) # [batch_size,n_heads,input_nums,attention_dim]
        atten_result = atten_result.transpose(1, 2).reshape(batch_size,-1,model_parameters.attention_heads*model_parameters.attention_dim) # 重新塑造最后一维，方便和投影矩阵进行相乘
        atten_result = weight_o(atten_result)
        return atten_scores,atten_result
    
    
class MultiHeadsAttentionLayer(nn.Module):
    def __init__(self):
        super(MultiHeadsAttentionLayer,self).__init__()
        # 后面把attention计算改成先算再拆分
        self.weight_k = nn.Linear(model_parameters.hidden_size,model_parameters.attention_dim*model_parameters.attention_heads,bias=False)
        self.weight_q = nn.Linear(model_parameters.hidden_size,model_parameters.attention_dim*model_parameters.attention_heads,bias=False)
        self.weight_v = nn.Linear(model_parameters.hidden_size,model_parameters.attention_dim*model_parameters.attention_heads,bias=False)
        self.weight_o = nn.Linear(model_parameters.attention_heads * model_parameters.attention_dim, model_parameters.hidden_size,bias=False)
        self.LN = nn.LayerNorm(model_parameters.hidden_size)
        # 后期把这个换成deepnorm
        
    def forward(self,input_vecs,atten_masks):
        # 残差
        residual, batch_size = input_vecs, input_vecs.size(0)
        # 计算过程：[batch_size,input_nums,input_dimention] -> [batch_zie,input_nums,attention_dimention*attention_heads]
        # ->split->(batch_size,input_nums,attention_heads,attention_dim) -> transpose->(batch_size,attention_heads,input_nums,attention_dim)
        Q = self.weight_q(input_vecs).view(batch_size, -1, model_parameters.attention_heads, model_parameters.attention_dim).transpose(1, 2) 
        K = self.weight_k(input_vecs).view(batch_size, -1, model_parameters.attention_heads, model_parameters.attention_dim).transpose(1, 2) 
        V = self.weight_v(input_vecs).view(batch_size, -1, model_parameters.attention_heads, model_parameters.attention_dim).transpose(1, 2) 
    
        # 拓展attention_musk
        atten_masks = atten_masks.unsqueeze(1).repeat(1,model_parameters.attention_heads,1,1) #在第二维上加一维，并且在第二维上重复注意力头数
        
        atten_scores,atten_result = GetScaledAttenScore()(Q, K, V,self.weight_o,atten_masks,batch_size)
        # 返回attention的post_LN结果
        return self.LN(atten_result + residual), atten_scores
    
    
class FullyConnectedNetwork(nn.Module):
    def __init__(self):
        super(FullyConnectedNetwork,self).__init__()
        self.fnn = nn.Sequential(
            nn.Linear(model_parameters.hidden_size,model_parameters.fnn_dim,bias=False),
            nn.ReLU(),
            nn.Linear(model_parameters.fnn_dim,model_parameters.hidden_size,bias=False)            
        )
        self.LN = nn.LayerNorm(model_parameters.hidden_size)
        
    def forward(self,input_vecs):
        residual = input_vecs
        output = self.fnn(input_vecs)
        return self.LN(residual+output)
    
    
class DecodeLayer(nn.Module):
    def __init__(self):
        super(DecodeLayer,self).__init__()
        self.atten = MultiHeadsAttentionLayer()
        # self.encoder_atten = MultiHeadsAttentionLayer()
        self.fnn = FullyConnectedNetwork()
        
    def forward(self,input_vecs,atten_masks):
        atten_result, atten_scores = self.atten(input_vecs,atten_masks)
        output = self.fnn(atten_result)
        return output, atten_scores
    
    
class Decoder(nn.Module):
    def __init__(self):
        # __init__作用，定义一个词嵌入层，一个位置嵌入层，以及六个dncoder layer
        super(Decoder,self).__init__()
        self.word_emb = nn.Embedding(model_parameters.vocab_size,model_parameters.hidden_size)
        self.pos_emb = nn.Embedding(model_parameters.max_pos,model_parameters.hidden_size) # 可以短于，不可以超过
        self.decode_layers = nn.ModuleList([DecodeLayer() for _ in range(model_parameters.n_layers)])
    
    def forward(self,input_vecs):
        '''生成一个和input_vecs相同维度的pos变量
        将嵌入后的词向量和位置向量相加作为输入
        获得attention的pad_mask以及subsequence_mask,并且通过torch.gt生成一个bool型的矩阵
        '''
        input_len = input_vecs.size(1)
        pos = torch.arange(input_len,dtype=torch.long, device=device) # 生成一个顺序序列
        # 先加维度然后才能expand
        pos = pos.unsqueeze(0).expand_as(input_vecs)
        pos_emb = self.pos_emb(pos)
        word_emb = self.word_emb(input_vecs)
        output_vecs = word_emb + pos_emb # 得到batch输入的embedding结果，维度为[batch_size,input_nums,hidden_size]
        atten_pad_mask = get_atten_pad_mask(input_vecs)
        try:
            atten_subsequence_mask = get_attn_subsequence_mask(input_vecs)
            atten_mask = torch.gt((atten_pad_mask+atten_subsequence_mask),0)
        except:
            atten_mask = torch.gt((atten_pad_mask),0) # 维度为[batch_size,input_nums.input_nums]
        
        # 然后遍历layers,获得fnn的输出
        attentions = []
        for layer in self.decode_layers:
            output_vecs, atten_socre = layer(output_vecs,atten_mask)
            attentions.append(atten_socre)
        
        return output_vecs,attentions
    
    
class GPT(nn.Module):
    def __init__(self):
        super(GPT,self).__init__()
        self.decoder = Decoder()
        self.projection = nn.Linear(model_parameters.hidden_size,model_parameters.vocab_size)
        print('the gpt2 model has been loaded')
        
    def forward(self,input_vecs):
        output_vecs, attentions = self.decoder(input_vecs)
        output_vecs = self.projection(output_vecs)
        return output_vecs.view(-1,output_vecs.size(-1)), attentions
    
    def auto_regressive_gen(self,input_vecs,generation_parameters):
        terminal = False
        is_begin = True
        prob_tree = utils.ProbTreeNode(prob=1,idx=0)
        start_len = len(input_vecs[0])
        beam_search_window = []
        node_lists = []
        max_new_tokens = model_parameters.max_new_tokens if model_parameters.max_new_tokens else 100
        while not terminal:
            do_or_not = random.random()
            if len(input_vecs[0] - start_len) > max_new_tokens:
                # input_vecs = utils.cat_answer_tensor(input_vecs,[next_symbol,word2id['<sep>']],device=device)
                input_vecs = utils.cat_answer_tensor(input_vecs,[word2id['<sep>']],device=device)
                break    
            output_vecs, _ = self.decoder(input_vecs) # 这里其实已经成tensor了
            projected_vecs = self.projection(output_vecs)
            if generation_parameters.beam_search:
                terminal,is_begin,prob_tree,input_vecs,node_lists,beam_search_window = utils.start_beam_search(
                    is_begin,
                    input_vecs,
                    projected_vecs,
                    prob_tree,
                    self.decoder,
                    self.projection,
                    beam_search_window,
                    node_lists,
                    device)
                continue
            
            elif generation_parameters.top_p is not None:
                if do_or_not < 0.5:
                    projected_vecs = projected_vecs.squeeze(0)
                    prob = projected_vecs.max(dim=-1, keepdim=False)[1]
                    next_word = prob.data[-1]
                    next_symbol = next_word
                else:
                    next_symbol = utils.top_p(projected_vecs, generation_parameters.top_p)
            
            elif generation_parameters.top_k is not None:
                _, next_symbols = projected_vecs.squeeze(0)[-1].topk(generation_parameters.top_k,dim=-1)
                if do_or_not < 0.5:
                    next_symbol = next_symbols[0]
                else:
                    next_symbol = random.choice(next_symbols.tolist())
            
            else:
                projected_vecs = projected_vecs.squeeze(0)
                prob = projected_vecs.max(dim=-1, keepdim=False)[1]
                next_word = prob.data[-1]
                next_symbol = next_word
            if next_symbol == word2id['<sep>']:
                terminal=True
            # torch.cat要求两个张量在维数上相同，且在拼接维度上一致，否则将报错
            # tensor.detach()函数用于返回一个新的张量，该张量与原始张量共享存储空间，但是不会被计算图追踪，也就是不会影响反向传播。
            input_vecs = utils.cat_answer_tensor(input_vecs,[next_symbol],device=device)

        return input_vecs
    
    def answer(self,sentence,generation_parameters):
        input_vecs = [word2id.get(word,1) if word != '\t' else word2id["<sep>"] for word in sentence]
        input_vecs = torch.tensor(input_vecs,device=device,dtype=torch.long).unsqueeze(0) # dim:[1,word_num]，之所以要这样干，是因为要和使用batch的情况统一
        output = self.auto_regressive_gen(input_vecs,generation_parameters).squeeze(0)
        out = [id2word[int(id)] for id in output]
        sep_index = []
        for idx,word in enumerate(out):
            if word == '<sep>':
                sep_index.append(idx)
        
        answer = ''.join(out[sep_index[-2]+1:-1])
        return answer
    
    
if __name__ == "__main__":
    class GeneratePara(BaseModel):
        beam_search: bool = False
        no_repeat_ngram: Union[None,int] = None
        top_p: Union[None,float] = None
        top_k: Union[None,int] = None
    gen_para = GeneratePara()
    gen_para.beam_search = True
    gen_para.top_p = 0.0005
    # gen_para.no_repeat_ngram = 4
    gpt_model = GPT().to('cuda')
    # gpt_model = GPT()
    gpt_model.load_state_dict(torch.load('/home/yuanyu/dev/hehaonan/gpt2-model/checkpoint/gpt2_qkv_right_4.0.pt'))
    answer = gpt_model.answer('一起去北京玩吧\t',generation_parameters=gen_para)
    print(answer)
