import torch
import numpy as np
from typing import Any, List, Union, Tuple
import random


class ProbTreeNode():
    ''''
    用于进行beam_search的树结构
    '''
    def __init__(self, prob:float, idx:int, children:Union[List, None]=None):
        self.prob = prob
        self.idx = idx
        self.children = children if children else []
        
    def pre_traval(self,node=None):
        if node is None:
            node = self
        yield node
        for child in node.children:
            yield from self.pre_traval(child)
            
    def __getitem__(self,idx):
        try:
            for _ in range(idx):
                node = next(iter(self.pre_traval()))
        except:
            print('索引超出结点数')
        return node

    def get_height(self,is_efficient:Union[bool,None]=None):
        # 求树的高度
        if not self.children:
            return 1
        else:
            if is_efficient:
                children_height = map(lambda x: x.get_height(), self.children[0])
            else:
                children_height = map(lambda x: x.get_height(), self.children)
            return max(list(children_height)) + 1
    
    def set_idx(self):
        # 设置整棵树的节点的索引，每个结点的索引是父节点的索引的三倍加自己的索引值
        for child in self.children:
            child.idx = (self.idx * 3) + child.idx
        for child in self.children:
            child.set_idx()
    
    def get_max_prob_product(self):
        # 求树上的最大概率乘积链
        # 在使用这个方法之前，要先使用set_idx方法
        if not self.children:
            return self.prob, self.idx
        else:
            max_product = float('-inf') # 初始化为无穷大
            max_idx = None
            for child in self.children:
                product, idx = child.get_max_prob_product() # 获得每颗子树的最大概率链，以及子树上最大链截止处的叶子节点的idx
                product *= self.prob
                if product > max_product:
                    max_product = product
                    max_idx = idx
            return max_product, max_idx

def build_beam_search_tree(window_length: int,projected_vecs: torch.Tensor,parent_node):
    # 构建概率链树
    current_n_prob = projected_vecs.squeeze(0) # 把大小为[batch_size,tokens,vocab_size]的输出向量的0维去掉，因为batch大小为1
    current_n_prob_values, current_n_prob_indexes = current_n_prob[-1].topk(window_length,dim=-1) # 只用关注最后一个预测的几大可能即可
    prob_node_list = []
    for idx, prob in enumerate(current_n_prob_values.tolist()):
        prob_node = ProbTreeNode(prob,idx) # 创建节点，prob中存概率值，idx为相对索引，需要在set_idx中绝对化
        prob_node_list.append(prob_node) # 顺序加入临时列表
    parent_node.children = prob_node_list # 放入父节点的孩子中
    return parent_node, current_n_prob_indexes, prob_node_list

def start_beam_search(is_begin,input_vecs,projected_vecs,prob_tree,decoder,projection,beam_search_window,node_lists,device):
    if is_begin:
        # 如果是一轮beam_search开始，则需要保存概率树的根节点
        prob_tree, beam_search_window,node_lists = build_beam_search_tree(3,projected_vecs,prob_tree)
        beam_search_window = [cat_answer_tensor(input_vecs,[i],device) for i in beam_search_window]
        is_begin = False
        return False,is_begin,prob_tree,input_vecs,node_lists,beam_search_window
    
    elif prob_tree.get_height() == 4:
        # 当概率树增长到一定高度，则直接截断，避免计算量过大
        terminal = False
        prob_tree.set_idx()
        max_prob_product, child_idx = prob_tree.get_max_prob_product()
        print(f"最大概率链的概率是{max_prob_product}")
        input_vecs = beam_search_window[child_idx] # 取出概率树中概率乘积最大的根节点，这就是我们需要的
        new_result = input_vecs.squeeze(0).tolist()
        beam_search_window = []
        is_begin = True
        prob_tree = ProbTreeNode(prob=1,idx=0)
        return terminal,is_begin,prob_tree,input_vecs,node_lists,beam_search_window
    
    else:
        terminal = False
        temp_beam_search_window = []
        temp_node_lists = []
        for node_content,node in zip(beam_search_window,node_lists): #这里还是tensor
            node_input_vecs,_ = decoder(node_content)
            node_projected_vecs = projection(node_input_vecs)
            node,beam_search_window,node_lists = build_beam_search_tree(3,node_projected_vecs,node)
            temp_node_lists.extend(node_lists)
            print(beam_search_window)
            current_result_list  = [cat_answer_tensor(node_content,[i],device) for i in beam_search_window]
            temp_beam_search_window.extend(current_result_list)
        beam_search_window = temp_beam_search_window
        node_lists = temp_node_lists
        return terminal,is_begin,prob_tree,input_vecs,node_lists,beam_search_window

def top_p(projected_vec: torch.Tensor, top_p = 0.1):
    current_n_prob = projected_vec.squeeze(0)
    _, current_n_indexes = current_n_prob[-1].sort(descending=True)
    current_n_indexes = current_n_indexes[:int(current_n_indexes.shape[-1] * top_p)]
    next_sambol = random.choice(current_n_indexes.tolist())
    return next_sambol

def cat_answer_tensor(input_vecs,new_vecs,device):
    return torch.cat([input_vecs.detach(),torch.tensor([new_vecs], dtype = input_vecs.dtype, device=device)],-1)
