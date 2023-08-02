import torch
import torch.nn as nn

from model import *

class LoRaAttentionLayer(nn.Module):
    def __init__(self,rank,attention_layer):
        super(LoRaAttentionLayer,self).__init__()
        self.rank = rank # lora模型的秩
        self.attention_layer = attention_layer # 原模型的attention
        self.weight_k = self.attention_layer.weight_k # 原模型的qkvo权重
        self.weight_q = self.attention_layer.weight_q
        self.weight_v = self.attention_layer.weight_v
        self.weight_o = self.attention_layer.weight_o
        self.LN = self.attention_layer.LN # 原模型的layerNorm层
        self.qa_metrix = nn.Linear(model_parameters.hidden_size,self.rank,bias=False) # Q = Q + AB
        self.ka_metrix = nn.Linear(model_parameters.hidden_size,self.rank,bias=False) # 论文中写的是Q = Q + BA，但是我觉得AB符合习惯一点
        self.va_metrix = nn.Linear(model_parameters.hidden_size,self.rank,bias=False)
        self.qb_metrix = nn.Linear(self.rank,model_parameters.attention_dim*model_parameters.attention_heads,bias=False)
        self.kb_metrix = nn.Linear(self.rank,model_parameters.attention_dim*model_parameters.attention_heads,bias=False)
        self.vb_metrix = nn.Linear(self.rank,model_parameters.attention_dim*model_parameters.attention_heads,bias=False)
        nn.init.zeros_(self.qa_metrix.weight) # A矩阵使用0初始化，B矩阵使用高斯分布初始化
        nn.init.zeros_(self.ka_metrix.weight)
        nn.init.zeros_(self.va_metrix.weight)

    def forward(self,input_vecs,atten_masks):
        residual, batch_size = input_vecs, input_vecs.size(0)
        Q = self.weight_q(input_vecs) + self.qb_metrix(self.qa_metrix(input_vecs))
        K = self.weight_k(input_vecs) + self.kb_metrix(self.ka_metrix(input_vecs))
        V = self.weight_v(input_vecs) + self.vb_metrix(self.va_metrix(input_vecs)) 
        Q = Q.view(batch_size,-1,model_parameters.attention_heads,model_parameters.attention_dim).transpose(1, 2)
        K = K.view(batch_size,-1,model_parameters.attention_heads,model_parameters.attention_dim).transpose(1, 2)
        V = V.view(batch_size,-1,model_parameters.attention_heads,model_parameters.attention_dim).transpose(1, 2)

    
        # 拓展attention_musk
        atten_masks = atten_masks.unsqueeze(1).repeat(1,model_parameters.attention_heads,1,1) #在第二维上加一维，并且在第二维上重复注意力头数
        
        atten_scores,atten_result = GetScaledAttenScore()(Q, K, V,self.weight_o,atten_masks,batch_size)
        # 返回attention的post_LN结果
        return self.LN(atten_result + residual), atten_scores
    
    
class LoraDecoderLayer(nn.Module):
    def __init__(self,rank,decoder_layer):
        super(LoraDecoderLayer,self).__init__()
        self.rank = rank
        self.fnn = decoder_layer.fnn
        self.atten = LoRaAttentionLayer(self.rank,decoder_layer.atten)

    def forward(self,input_vecs,atten_masks):
        atten_result, atten_scores = self.atten(input_vecs,atten_masks)
        output = self.fnn(atten_result)
        return output, atten_scores
        
    
class LoraDecoder(nn.Module):
    def __init__(self,rank,decoder):
        # __init__作用，定义一个词嵌入层，一个位置嵌入层，以及六个dncoder layer
        super(LoraDecoder,self).__init__()
        self.rank = rank
        self.word_emb = decoder.word_emb
        self.pos_emb = decoder.pos_emb # 可以短于，不可以超过
        self.decode_layers = decoder.decode_layers
        
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
            lora_decoder_layer = LoraDecoderLayer(self.rank,layer)
            output_vecs, atten_socre = lora_decoder_layer(output_vecs,atten_mask)
            attentions.append(atten_socre)
        
        return output_vecs,attentions
    
    
class LoRaModel(nn.Module):
    def __init__(self,rank,path):
        super(LoRaModel,self).__init__()
        self.total_param = 0
        self.rank = rank
        self.gpt_model = GPT()
        if path:
            self.gpt_model.load_state_dict(torch.load(path))
            print('已加载预训练权重')
        else:
            print('不加载预训练权重')
        for parameter in self.gpt_model.parameters():
            parameter.requires_grad = False
        self.lora_decoder = LoraDecoder(self.rank,self.gpt_model.decoder)
        self.projection = self.gpt_model.projection
        
    def is_freeze_weight(self):
        for name, param in self.gpt_model.named_parameters():
            print(name, param.requires_grad)
            self.total_param += param.numel()
        print(f'模型总参数量为{round(self.total_param / (1024*1024),2)}M')
    
    def forward(self,input_vecs):
        output_vecs, attentions = self.lora_decoder(input_vecs)
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
            output_vecs, _ = self.lora_decoder(input_vecs) # 这里其实已经成tensor了
            projected_vecs = self.projection(output_vecs)
            if generation_parameters.beam_search:
                terminal,is_begin,prob_tree,input_vecs,node_lists,beam_search_window = utils.beam_search(
                    is_begin,
                    input_vecs,
                    projected_vecs,
                    prob_tree,
                    self.lora_decoder,
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
    gen_para.beam_search = False
    # gen_para.top_p = 0.0005
    # gen_para.no_repeat_ngram = 4
    # gpt_model = GPT().to('cuda')
    lora_model = LoRaModel(8)
    answer = lora_model.answer('你吃饭了吗\t',generation_parameters=gen_para)
    print(answer)
