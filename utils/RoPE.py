import torch

from typing import Union, Tuple
"""
作者: hhn
本代码实现借鉴于llama,地址:https://github.com/facebookresearch/llama/blob/main/llama/model.py#L131, 我在llama的实现上稍作修改使其有更好的易读性。
RoPE论文地址:https://arxiv.org/abs/2104.09864, 博客地址:https://spaces.ac.cn/archives/8265/comment-page-3#comments

RoPE:对于q,k向量,我们需要一个f(x,pos)=x_pos_emb[其中x为任意q,k向量],使得<f(q,m)f(k,n)> == g(q,k,m-n)。
其中<vecs1vecs2>表示两个向量求内积,m与n分别为q和k在输入序列中的位置,m-n为他们的相对位置。
推导可以知道有一个f(q,m) = qe**(m(theta_i)i)满足f(q,m)f(k,n) == g(q,k,m-n),其中theta_i为和i有关的常量
theta_i == 10000.0**(-2i/d), 其中i为q向量中元素的位置,d为q向量的长度,也就是attention的维度

当把q看作复平面上的复向量,则f的作用相当于令q和1e**(m(theta_i)i) == cos(m(theta_i)) + sin(m(theta_i))i相乘
从复向量的角度来说相当于是把q旋转了一定的角度,所以我将1e**(m(theta_i)i)称为旋转向量
"""
THETA = 10000.0

def get_rotate_vecs(
    atten_dim: int,
    seq_len: int,
    theta: Union[None,int]=None
) -> torch.Tensor: 
    """
    作用：求旋转向量
    输入: atten_dim->注意力q/k向量的维度, seq_len->输入序列的长度
    输出：旋转向量
    """
    if theta is None:
        theta = THETA
    # 一个长度atten_dim // 2的向量，每个位置为i/64，其中i=0，2，4.....,这一步求的是(2i/d)
    theta_i_e = torch.arange(0,atten_dim,2)[:(atten_dim // 2)].float() / atten_dim 
    theta_i = theta ** theta_i_e 
    freqs = 1.0 / theta_i # 1/theta**(2i/d)不知道为什么是求倒数不是复数
    position = torch.arange(seq_len) # [0, 1, 2, 3, ..., seq_len] 
    freqs = torch.outer(position,freqs).float() # 求向量的外积,维度为[seq_len,atten_dim]
    freqs_cis = torch.polar(torch.ones_like(freqs),freqs) #将上一步的结果写成复数的形式,模是1幅角是freqs
    return freqs_cis.view(1,seq_len,1,atten_dim//2)

def get_origin_rotate_vecs(
    atten_dim: int,
    seq_len: int,
    theta: Union[None,int]=None
) -> torch.Tensor: 
    # 论文中的rope实现方式
    if theta is None:
        theta = THETA
    theta_i_e = (torch.arange(0,atten_dim,2)[:(atten_dim // 2)].float() / atten_dim) * -1
    freqs = theta_i = theta ** theta_i_e
    position = torch.arange(seq_len) # [0, 1, 2, 3, ..., seq_len] 
    freqs = torch.outer(position,freqs).float() # 求向量的外积,维度为[seq_len,atten_dim] 
    freqs_cis = torch.polar(torch.ones_like(freqs),freqs) #将上一步的结果写成复数的形式,模是1幅角是freqs
    return freqs_cis.view(1,seq_len,1,atten_dim//2)
     

def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rotate_vecs: torch.Tensor
) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """
    作用: 将q,k向量分别与旋转向量相乘,得到旋转后的q,k向量q/k_rotated。然后进行点乘得到具有位置信息的attention分数
    输入: q->weight_q(input_vecs), k->weight_k(input_vecs), rotaed_vecs->旋转向量
    输出: 注意力分数
    """
    # 计算过程q:[batch_size,seq_len,atten_heads,atten_dim]->q_complex:[b,s,a_h,a_d//2,2]->[b,s,a_h,a_d//2]->[b,a,a_h,a_d//2,2]
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2)) #[batch_size,seq_len,atten_heads,atten_dim//2,2]
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2)) # 将一个大小为n的向量两两组合形成复数来计算
    # 位置编码只和向量的序列位置还有向量本身有关，和batch以及注意力头无关，所以只用关注第二维和第四维
    q_rotated = torch.view_as_real(q_complex*rotate_vecs).flatten(3) # 恢复成原来的样子，将第三维之后压平，也就是(atten_dim//2,2)->(atten_dim)
    k_rotated = torch.view_as_real(k_complex*rotate_vecs).flatten(3)
    attention_score = torch.matmul(q_rotated, k_rotated.transpose(-1,-2))
    return q_rotated.type_as(q), k_rotated.type_as(q), attention_score.type_as(q)
    

if __name__ == '__main__':
    # 代码的示例使用，直接运行可以看到示例输出
    atten_dim = 64
    atten_heads = 8
    batch_size = 1
    seq_len = 4
    q = torch.rand(batch_size,seq_len,atten_heads,atten_dim)
    k = torch.rand(batch_size,seq_len,atten_heads,atten_dim)
    rotate_vecs = get_origin_rotate_vecs(seq_len=seq_len,atten_dim=atten_dim)
    q_rotated, k_rotated, atten_score = apply_rope(q,k,rotate_vecs)
    print(atten_score)
