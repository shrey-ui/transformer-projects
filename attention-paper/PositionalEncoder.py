
import torch.nn as nn
import torch
import math
import numpy as np 

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


