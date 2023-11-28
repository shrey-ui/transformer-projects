#from attention-paper.MultiHeadAttention import self_attention_unit

import torch.nn as nn
import torch
import numpy as np 
import math 
import torch.nn.functional as F

def selfattention(q, k, v, mask= None):
    d_k= q.size()[-1]
    scaled= torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        scaled+= mask

    attention= F.softmax(scaled, dim= -1)
    values= torch.matmul(attention, v)

    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, d_model, mask= None):
        super(MultiHeadAttention, self).__init__()
        self.input_dim= input_dim
        self.num_heads= num_heads
        self.d_model= d_model 
        self.qkv_layer= nn.Linear(self.d_model, 3*self.d_model)
        self.linear_layer= nn.Linear(d_model, d_model)

        self.mask= mask

    def forward(self, x, mask):

        batch_size, seq_length, _ = x.size()
        qkv= self.qkv_layer(x)
        print(qkv.shape)
        qkv= qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.d_model//self.num_heads)
        qkv= qkv.permute(0, 2, 1, 3)
        q, k, v= torch.chunk(qkv, chunks= 3, dim=-1)

        values, attention= selfattention(q, k, v, mask= mask)
        
        values= values.permute(0, 2, 1, 3)
        print(values.shape)
        values= values.reshape(batch_size, seq_length, self.d_model)
        
        output= self.linear_layer(values)

        return output


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_sequence_len):
        super(PositionalEncoder, self).__init__()
        self.d_model= d_model
        self.max_len= max_sequence_len

        self.eveni= torch.arange(0, d_model, 2).float()
        self.oddi= torch.arange(1, d_model, 2).float()

        self.denominator= torch.pow(10000, self.eveni/d_model)
        self.position= torch.arange(self.max_len, dtype= torch.float,
                                    ).reshape(self.max_len ,1)

    def forward(self):
        even_pe= torch.sin(self.position/self.denominator)
        odd_pe= torch.cos(self.position/self.denominator)

        embedding = torch.stack([even_pe, odd_pe], dim= 2)
        embed_final= torch.flatten(embedding, start_dim= 1, end_dim=2)

        return embed_final


class LayerNormalization(nn.Module):
    def __init__(self, parameter_shape, eps= 1e-5):
        super(LayerNormalization, self).__init__()
        self.parameter_shape= parameter_shape
        self.eps= eps
        self.gamma= nn.Parameter(torch.ones(self.parameter_shape))
        self.beta= nn.Parameter(torch.zeros(self.parameter_shape))

    def forward(self, inputs):
        dimensions= [-(i+1) for i in range(len(self.parameter_shape))]
        mean= torch.mean(inputs, dim= dimensions, keepdim= True)
        var= ((inputs- mean) **2).mean(dim= dimensions, keepdim= True)

        std= (var + self.eps).sqrt()

        y= (inputs-mean)/std

        out= self.gamma * y  + self.beta
        

        return out

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, hidden):
        super(FeedForwardLayer, self).__init__()
        self.d_model= d_model
        self.hidden= hidden

        self.fc1 = nn.Linear(self.d_model, self.hidden)
        self.fc2= nn.Linear(self.hidden, self.d_model)
        
        self.relu= nn.ReLU()
        self.dropout= nn.Dropout(p= 0.3)
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, input_dim, max_seq_len, mask= None):
        super(EncoderLayer, self).__init__()
        self.d_model =d_model 
        self.num_heads= num_heads
        self.input_dim =input_dim
        self.max_seq_len= max_seq_len
        self.mask= mask
        self.posencoder= PositionalEncoder(self.d_model, self.max_seq_len)
        self.layernorm= LayerNormalization([d_model])
        self.multiheadattn= MultiHeadAttention(self.input_dim, self.num_heads, self.d_model)
        self.layernorm1= LayerNormalization([d_model])
        self.feedforward_lay= FeedForwardLayer(self.d_model, 256)

    def forward(self, x, attention_mask= None):
        residual_x= x.clone()
        
        attn_output= self.multiheadattn(x, attention_mask)
        attn_output= attn_output + residual_x        # skip connection
        ln_output= self.layernorm(attn_output)
        residual_x= ln_output.clone()
        ff_out= self.feedforward_lay(ln_output)
        ff_out= ff_out + residual_x         #skip connection 2
        output= self.layernorm1(ff_out)
        

        return output

class SequentialEncoder(nn.Sequential):
    def __init__(self, *inputs):
        x= inputs
        for module in self._modules.values():
            x= module(x, self_attention_mask)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, 
                 input_dim, num_heads,
                 max_seq_len, mask= None):

        super(Encoder, self).__init__()
        self.d_model= d_model
        self.num_layers= num_layers
        self.num_heads= num_heads
        self.max_seq_len= max_seq_len
        self.mask =mask
        self.input_dim =input_dim

        #self.encoder= EncoderLayer(self.d_model, self.num_heads, self.input_dim,
         #@                          self.max_seq_len)

        self.layers= nn.Sequential(*[EncoderLayer(d_model,
                                                      num_heads, 
                                                      input_dim, 
                                                      max_seq_len) 
                                       for _ in range(self.num_layers)])
        
    def forward(self, x):
        x= self.layers(x)
        return x


if __name__ == "__main__":
    pe= PositionalEncoder(d_model =6, max_sequence_len= 10)
    print(pe.forward())

    inputs= torch.randn(5, 3, 8)
    layer_norm = LayerNormalization(inputs.size()[-1:])
    print(layer_norm.forward(inputs))

    encoder= Encoder(d_model = 512, 
                     num_layers= 6, 
                     input_dim = 512, 
                     num_heads= 8, 
                     max_seq_len = 200,
                     mask= None
                     )

    x= torch.randn(1, 4, 512)
    self_attention_mask = torch.zeros((1, 8, 4, 4))
    print(encoder.forward(x))

    











