import json
import torch
import time
import sys
import traceback
import ast
import re

from model import *

from tqdm import tqdm
from torch import nn, optim

import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data

def make_data(datas):
    train_datas = []
    for data in datas:
        data  = data.strip()
        train_data = [i if i!='\t' else '<sep>' for i in data] + ['<sep>']
        train_datas.append(train_data)
    return train_datas

class TrainDataSet(Data.Dataset):
    def __init__(self,datas):
        self.datas = datas
        
    def __getitem__(self,item):
        '''
        输入：某一项
        输出：force_teach数据以及长度
        '''
        data = self.datas[item]
        if len(data) > model_parameters.max_pos:
            data  = data[:model_parameters.max_pos]
        ft_input = data[:-1]
        ft_output = data[1:]
        
        ft_input_len = len(ft_input)
        ft_output_len = len(ft_output)
        
        return {'ft_input':ft_input,'ft_output':ft_output,'ft_input_len':ft_input_len,'ft_output_len':ft_output_len}
    
    def __len__(self):
        '''
        作用：返回对话数据集的长度，也就是对话数量
        '''
        return len(self.datas)
    
    def pad_batch_data(self,batch):
        max_input_len = max([i['ft_input_len'] for i in batch])
        max_output_len = max([i['ft_output_len'] for i in batch])
        
        for data in batch:
            data['ft_input'].extend([word2id['<pad>']]*(max_input_len - data['ft_input_len']))
            data['ft_output'].extend([word2id['<pad>']]*(max_output_len - data['ft_output_len']))
        ft_inputs = torch.tensor([i['ft_input'] for i in batch], device=device, dtype= torch.long)
        ft_outputs = torch.tensor([i['ft_output'] for i in batch], device=device, dtype= torch.long)
        return ft_inputs,ft_outputs
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_epoch(model,data_loader,optimizer,criterion,clip=1,print_every=None):
    """
    1.将模型设置为训练模式
    2.对dataloader中的数据进行操作
    ·将梯度设置为0
    ·得到模型的输出和atten_socres
    ·计算loss
    ·反向传播
    """
    model.train()
    if print_every == 0: print_every = 1 # 分母不能是0
    print_every_loss = 0 
    epoch_loss = 0
    every_avg_loss_list = []
    for i , (ft_inputs,ft_outputs) in enumerate(tqdm(data_loader)):
        # if i>21815:
        try:
            # print('train a batch data')
            # sys.stdout.flush()
            optimizer.zero_grad()
            model_outputs, attentions = model(ft_inputs)
            
            loss = criterion(model_outputs,ft_outputs.view(-1)) # 将model_output展开为一维tensor
            print_every_loss+=loss.item() # 把这一个数据的loss加上去
            epoch_loss+=loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step()
            
            if print_every and (i+1)%print_every==0:
                print_loss_avg = print_every_loss / print_every
                print_every_loss = 0
                # print(f"the {i}th batch's average loss is {print_loss_avg}")
                every_avg_loss_list.append(print_loss_avg)  
                # sys.stderr.flush()
        except Exception:
            traceback.print_exc()
            print(ft_inputs)
            print(ft_outputs) 

    return epoch_loss/len(data_loader), every_avg_loss_list

def train(model,data_loader,learning_rate,epochs):
    '''
    1.设置损失函数
    2.设置优化器
    3.对每个epoch进行训练
    4.保存检查点
    '''
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device) # 忽略目标值0处的梯度
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, every_avg_loss_list = train_epoch(model,data_loader,optimizer,criterion,print_every=100)
        with open('loss_log.txt','a') as f:
            f.write(str(every_avg_loss_list))
        end_time = time.time()
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(),'checkpoint/gpt2_qkv_right_{}.pt'.format(str((epoch+1)/5) + 4))
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # sys.stderr.flush()
        print(f'\tTrain Loss: {train_loss:.3f}')
        # sys.stderr.flush()
        with open('./nohup.out','w') as f:
            f.write('')
        
        
        
if __name__ == '__main__':
    datas = pd.read_json('./train.json')['dialog'].to_list()
    train_data = make_data(datas)
    
    train_data = [[word2id[i] if i in word2id else 0 for i in line] for line in train_data]
    batch_size = 8
    epochs = 5
    lr = 1e-4
    dataset = TrainDataSet(train_data)
    data_loader = Data.DataLoader(dataset,batch_size=batch_size,collate_fn=dataset.pad_batch_data)
    model = GPT().to(device)
    model.load_state_dict(torch.load(''))
    
    train(model,data_loader,lr,epochs)        
    with open('loss_log.txt','r') as f:
        all_loss = re.findall(r'\[.*?\]',f.read())
    all_loss_list = []
    for i in  all_loss:
        all_loss_list.extend(ast.literal_eval(i))
    x = [i for i in range(len(all_loss_list))]
    plt.scatter(x, all_loss_list, s=1)  # s=1 使得散点的大小为1
    plt.title('Original Data')
    plt.xlabel('Index')
    plt.ylabel('Loss')
    plt.savefig('loss_image/loss.png')
    plt.show()

    smooth_data = np.array(all_loss_list).astype(float).flatten().tolist()
    smooth_data = np.convolve(smooth_data, np.ones(20)/20, mode='valid')
    plt.plot(smooth_data, color='red')
    plt.title('Smoothed Data')
    plt.xlabel('Index')
    plt.ylabel('Loss')
    plt.savefig('loss_image/smooth_loss.png')
    plt.show()
            
    
        
        
