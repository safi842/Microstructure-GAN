import os
import pandas as pd
import random
from collections import OrderedDict
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import make_grid

class ccbn(nn.Module):
    def __init__(self, input_size, output_size, eps=1e-4, momentum=0.1):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = spectral_norm(nn.Linear(input_size, output_size, bias = False), eps = 1e-4)
        self.bias = spectral_norm(nn.Linear(input_size, output_size, bias = False), eps = 1e-4)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        
        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var',  torch.ones(output_size))
    
    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
        return out * gain + bias
    
    def extra_repr(self):
        s = 'out: {output_size}, in: {input_size},'
        return s.format(**self.__dict__)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation = nn.ReLU(inplace = False)):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class GeneratorResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample = None, embed_dim = 128, dim_z = 128):
        super(GeneratorResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = self.in_channels // 4
        
        self.conv1 = spectral_norm(nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size = 1, padding = 0), eps = 1e-4)
        self.conv2 = spectral_norm(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, padding = 1), eps = 1e-4)
        self.conv3 = spectral_norm(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, padding = 1), eps = 1e-4)
        self.conv4 = spectral_norm(nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size = 1, padding = 0), eps = 1e-4)
        
        self.bn1 = ccbn(input_size = (3 * embed_dim) + dim_z, output_size = self.in_channels)
        self.bn2 = ccbn(input_size = (3 * embed_dim) + dim_z, output_size = self.hidden_channels)
        self.bn3 = ccbn(input_size = (3 * embed_dim) + dim_z, output_size = self.hidden_channels)
        self.bn4 = ccbn(input_size = (3 * embed_dim) + dim_z, output_size = self.hidden_channels)
        
        self.activation = nn.ReLU(inplace=False)
        
        self.upsample = upsample
        
    def forward(self,x,y):
        # Project down to channel ratio
        h = self.conv1(self.activation(self.bn1(x, y)))
        # Apply next BN-ReLU
        h = self.activation(self.bn2(h, y))
        # Drop channels in x if necessary
        if self.in_channels != self.out_channels:
            x = x[:, :self.out_channels]      
        # Upsample both h and x at this point  
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        # 3x3 convs
        h = self.conv2(h)
        h = self.conv3(self.activation(self.bn3(h, y)))
        # Final 1x1 conv
        h = self.conv4(self.activation(self.bn4(h, y)))
        return h + x

class Generator(nn.Module):
    def __init__(self, G_ch = 64, dim_z=128, bottom_width=4, img_channels = 1,
                 init = 'ortho',n_classes_temp = 7, n_classes_time = 8, n_classes_cool = 4, embed_dim = 128):
        super(Generator, self).__init__()
        self.ch = G_ch
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.init = init
        self.img_channels = img_channels

        self.embed_temp = nn.Embedding(n_classes_temp, embed_dim)
        self.embed_time = nn.Embedding(n_classes_time, embed_dim)
        self.embed_cool = nn.Embedding(n_classes_cool, embed_dim)
        
        self.linear = spectral_norm(nn.Linear(dim_z + (3 * embed_dim), 16 * self.ch * (self.bottom_width **2)), eps = 1e-4)
        
        self.blocks = nn.ModuleList([
                GeneratorResBlock(16*self.ch, 16*self.ch),
                GeneratorResBlock(16*self.ch, 16*self.ch, upsample =  nn.Upsample(scale_factor = 2)),
                GeneratorResBlock(16*self.ch, 16*self.ch),
                GeneratorResBlock(16*self.ch, 8*self.ch, upsample =  nn.Upsample(scale_factor = 2)),
                GeneratorResBlock(8*self.ch, 8*self.ch),
                GeneratorResBlock(8*self.ch, 8*self.ch, upsample =  nn.Upsample(scale_factor = 2)),
                GeneratorResBlock(8*self.ch, 8*self.ch),
                GeneratorResBlock(8*self.ch, 4*self.ch, upsample =  nn.Upsample(scale_factor = 2)),
                Self_Attn(4*self.ch),
                GeneratorResBlock(4*self.ch, 4*self.ch),
                GeneratorResBlock(4*self.ch, 2*self.ch, upsample =  nn.Upsample(scale_factor = 2)),
                GeneratorResBlock(2*self.ch, 2*self.ch),
                GeneratorResBlock(2*self.ch,  self.ch, upsample =  nn.Upsample(scale_factor = 2))
        ])
        
        self.final_layer = nn.Sequential(
                nn.BatchNorm2d(self.ch),
                nn.ReLU(inplace = False),
                spectral_norm(nn.Conv2d(self.ch, self.img_channels, kernel_size = 3, padding = 1)),
                nn.Tanh()
        )
        
        self.init_weights()
                                    
    def init_weights(self):
        print(f"Weight initialization : {self.init}")
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    torch.nn.init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    torch.nn.init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    torch.nn.init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print("Param count for G's initialized parameters: %d Million" % (self.param_count/1000000))
        
        
    def forward(self,z , y_temp, y_time, y_cool):
        y_temp = self.embed_temp(y_temp)
        y_time = self.embed_time(y_time)
        y_cool = self.embed_cool(y_cool)
        z = torch.cat([z, y_temp, y_time, y_cool], 1)     
        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)    
        # Loop over blocks
        for i, block in enumerate(self.blocks):
            if i != 8:
                h = block(h, z)
            else:
                h = block(h)
        # Apply batchnorm-relu-conv-tanh at output
        h = self.final_layer(h)
        return h