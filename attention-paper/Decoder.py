from sympy.functions.combinatorial.numbers import _MultisetHistogram
import torch
import torch.nn as nn
import numpy as np
import math
import PositionalEncoder as PE 
import LayerNormalization as LN
import torch.nn.functional as F
import encoder



def selfattention(q, k, v, mask):
    d_k= q.size()[-1]
    updated= torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        updated= updated + mask
    attention= F.softmax(updated, dim=-1)
    values= torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model= d_model
        self.num_heads= num_heads

        self.d_k= d_model//num_heads

        self.qkv_layer= nn.Linear(d_model, 3*d_model)
        self.linear_layer= nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.size()
        qkv= self.qkv_layer(x)

        qkv= qkv.reshape(batch_size, seq_len, self.num_heads, 3*self.d_k)
        qkv= torch.permute(qkv, dims= (0, 2, 1, 3))
        q, k, v= qkv.chunk(chunks= 3, dim= -1)

        values, attention= selfattention(q, k, v, mask)
        values= torch.permute(values, dims= (0, 2, 1, 3))
        values= values.reshape(batch_size, seq_len, self.d_model)
        values= self.linear_layer(values)
        return values

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, ):
        super(CrossAttention, self).__init__()
        self.d_model= d_model
        self.num_heads= num_heads

        self.head_dim= d_model//num_heads
        
        self.kv_layer= nn.Linear(d_model, 2*d_model)
        self.q_layer= nn.Linear(d_model, d_model)
    
        self.linear_layer= nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask):
        batch_size, seq_len, _= x.size()
        kv_input= self.kv_layer(x)
        kv_input= kv_input.reshape(batch_size, seq_len, self.num_heads, 2*self.head_dim)

        kv_input= kv_input.permute(0, 2, 1, 3)
        q_input= self.q_layer(y)
        q_input= q_input.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q_input= q_input.permute(0, 2, 1, 3)

        k, v= kv_input.chunk(chunks= 2, dim= -1)

        values, attention= selfattention(q_input, k, v, mask)
        values= values.permute(0, 2, 1, 3)

        values= values.reshape(batch_size, seq_len, self.d_model)
        
        values= self.linear_layer(values)
        
        return values

class FeedForward(nn.Module):

    def __init__(self, d_model, hidden):
        super(FeedForward, self).__init__()
        self.d_model= d_model
        self.hidden= hidden

        self.fc1 = nn.Linear(self.d_model, self.hidden)
        self.fc2= nn.Linear(self.hidden, self.d_model)
        
        self.relu= nn.ReLU()
        self.dropout= nn.Dropout(p= 0.3)
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads):
        

        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads= num_heads 
        #self.max_seq_len= self.max_seq_len

        self.masked_attention= MultiHeadAttention(self.d_model, self.num_heads)
        self.layer_norm_1= LN.LayerNormalization([d_model])
        self.drop_1= nn.Dropout(p= 0.3)
        #self.encoder= 
        self.cross_attention= CrossAttention(self.d_model, self.num_heads)

        self.layer_norm_2= LN.LayerNormalization([d_model])
        self.drop_2= nn.Dropout(p= 0.3)

        self.feedforward= FeedForward(self.d_model, 2*self.d_model) 
        
        self.layer_norm_3= LN.LayerNormalization([d_model])
        self.drop_3= nn.Dropout(p= 0.3)
    
    def forward(self, x, y, mask):
        
        y_clone= y.clone()
        y_updated= self.masked_attention.forward(y, mask)
        #y_updated= self.layer_norm_1.forward(y_updated)
        y_updated= self.drop_1(y_updated)
        y_att= y_updated+ y_clone
        y_att= self.layer_norm_1.forward(y_att)
        y_clone= y_att.clone()


        y_cross= self.cross_attention.forward(x, y, mask= None)
        y_cross= self.drop_2(y_cross)        
        y_cross= self.layer_norm_2.forward(y_cross)
        y_cross= y_cross+ y_clone
        

        y_clone= y_cross.clone()
        y_ff= self.feedforward.forward(y_cross)
        y_ff= self.drop_3(y_ff)
        y_ff= y_ff + y_clone
        y_ff= self.layer_norm_3.forward(y_ff)        
    

        return y_ff


class DecoderSequential(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask= inputs
        for module in self._modules.values():
            y= module(x, y, mask)
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, num_layers):
        
        super(Decoder, self).__init__()
        self.d_model= d_model
        self.num_heads= num_heads
        self.num_layers= num_layers
        self.max_seq_len= max_seq_len
        self.layers= DecoderSequential(*[DecoderLayer(self.d_model, self.num_heads) for _ in range(self.num_layers)])

    def forward(self, x, y, mask):
        y= self.layers(x, y, mask)

        return y
        
        


if __name__ == "__main__":
    input_Tens= torch.randn(1, 4, 512)
    d_model= 512
    num_heads= 8
    mask = torch.ones((4, 4)).tril()
    mask[mask==0] = -torch.inf
    print(mask)
    
    AttentionHead= Decoder(d_model, num_heads, 4, 3)
    values= AttentionHead.forward(input_Tens,input_Tens, mask= mask)
    print(values)


