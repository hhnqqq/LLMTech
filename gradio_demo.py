import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import torch 
import gradio as gr
import json
import argparse
import time

from pydantic import BaseModel
from typing import Union

from model import GPT
from model_qkv_right import GPT as qkv_GPT

class GeneratePara(BaseModel):
    beam_search: bool = False
    no_repeat_ngram: Union[None,int] = None
    top_p: Union[None,float] = None
    top_k: Union[None,int] = None
gen_para = GeneratePara()
gen_para.beam_search = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT().to('cuda')
model1 = qkv_GPT().to('cuda')
model.load_state_dict(torch.load())
model1.load_state_dict(torch.load())
dict_datas = json.load(open())
word2id, id2word = dict_datas['word2id'], dict_datas['id2word']

dialog1 = ''
dialog2 = ''

with gr.Blocks() as demo:
    gr.Markdown('# 这是我自己开发的gpt2闲聊模型')
    with gr.Row():
        chatbot1 = gr.Chatbot(label='聊天记录')
        chatbot2 = gr.Chatbot(label='聊天记录')
    msg = gr.Textbox(label='请输入你的信息')

    
    def clear_dialog():
        global dialog1,dialog2
        dialog1 = ''
        dialog2 = ''
         
    def to_block(prompt, chat_history):
        global dialog1,dialog2
        model.eval()
        if prompt != '清空':
            dialog1 += prompt + '\t'
            answer = model.answer(dialog1, gen_para)
            dialog1 += answer + '\t'
            print(dialog1)
            chat_history.append((prompt, answer))
        else:
            dialog1 = ''
            chat_history.append((prompt,'已清空聊天历史'))
        return '', chat_history
    
    def to_block1(prompt, chat_history):
        global dialog1,dialog2
        model.eval()
        if prompt != '清空':
            dialog2 += prompt + '\t'
            answer = model1.answer(dialog1, gen_para)
            dialog2 += answer + '\t'
            print(dialog2)
            chat_history.append((prompt, answer))
        else:
            dialog2 = ''
            chat_history.append((prompt,'已清空聊天历史'))
        return '', chat_history

    
    with gr.Row():
        submit_button = gr.Button("Submit")
        clear_button = gr.ClearButton([msg,chatbot1])
        submit_button.click(to_block, [msg,chatbot1], [msg,chatbot1])
        clear_button.click(clear_dialog)
    with gr.Row():
        submit_button = gr.Button("Submit")
        clear_button = gr.ClearButton([msg,chatbot2])
        submit_button.click(to_block, [msg,chatbot2], [msg,chatbot2])
        clear_button.click(clear_dialog)
    msg.submit(to_block, [msg,chatbot1], [msg,chatbot1])
    msg.submit(to_block1, [msg,chatbot2], [msg,chatbot2])

demo.launch(share=True,server_name='0.0.0.0')    
# print(to_black('你好'))
