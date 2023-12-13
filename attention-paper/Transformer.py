import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
from Decoder import Decoder 
import encoder
import LayerNormalization as LN 
import PositionalEncoder as PE 


class Transformer(nn.Module):
    def __init__(self, d_model, 
                 max_seq_len, 
                 num_heads, 
                 input_seq_length,
                 encoder_layers,
                 decoder_layers, 
                 drop_prob, 
                 ffn_hidden, output_size):  
        super(Transformer, self).__init__()
        self.output_size= output_size
        self.encoder= encoder.Encoder(d_model= d_model, 
                                      num_layers= encoder_layers, 
                                      num_heads= num_heads, 
                                      max_seq_len= max_seq_len, 
                                      input_dim = d_model
                                      )

        self.decoder= Decoder(d_model= d_model, num_heads= num_heads, 
                              max_seq_len= max_seq_len, 
                              num_layers= decoder_layers,)

        self.linear= nn.Linear(d_model, self.output_size)

    def forward(self, x, y, mask_encoder, mask_decoder, ):
        x= self.encoder.forward(x)
        out= self.decoder.forward(x, y, mask_decoder)
        out= self.linear(out)
        return out


if __name__ == "__main__" : 
    input_examp= torch.randn(1, 
                             100, 
                             512)
    
    transformer= Transformer(512, 
                             100, 
                             8, 
                             512, 
                             3, 
                             4, 
                             0.3, 
                             256, 
                             1024, 
                             )

    mask_examp= torch.ones(100, 100).tril()
    mask_examp[mask_examp==0]= -torch.inf
    print(mask_examp)

    print(transformer.forward(input_examp, input_examp, torch.zeros(100, 100), mask_examp))



