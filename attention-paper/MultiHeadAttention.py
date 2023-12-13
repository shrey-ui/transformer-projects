import torch
import torch.nn as nn
import numpy as np

def self_attention_unit(q, k, v, mask None):
    d_k= q.size()[-1]
    scaled= torch.matmul(q, k.transpose(-2, -1))//math.sqrt(d_k)
    if mask is not None:
        scaled= scaled + mask

    attention= nn.Softmax(attention, dim= -1)
    values= torch.matmul(attention, v)
    return attention, values

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads,input_dim, mask= None):

        self.d_model= d_model
        self.num_heads= num_heads
        self.input_dim= input_dim
        self.qkv_layer= nn.Linear(self.input_dim, 3*self.input_dim)
        self.linear_layer= nn.Linear(d_model, d_model)
        self.softmax= nn.Softmax(dim= 1)
        self.mask= mask

    def forward(self, x):
        qkv= self.qkv_layer(x)
        qkv_reshape= torch.reshape(qkv, (qkv.shape[0], qkv.shape[1], self.num_heads, torch.shape[2]//self.num_heads)                          
        qkv_reshape= qkv_reshape.permute(0, -2, 1, -1)
        
        q, k, v= qkv_reshape.chunk(3, dim=-1)

        attention, values= self_attention_unit(q, k, v, mask= self.mask)

        values= values.reshape(qkv.shape[0], qkv.shape[1], self.num_heads*values.shape()[-1])
        values= self.linear_layer(values)
        return attention, values
