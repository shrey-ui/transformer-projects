
import torch.nn as nn
import torch
import numpy as np


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
