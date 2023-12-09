import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from src.model.SincConv import SincConv_fast

class FMS(nn.Module):
    def __init__(self, filter_length: int):
        super(FMS, self).__init__()
        self.avpool = nn.AdaptiveAvgPool1d(1)
        self.model = nn.Sequential(
            nn.Linear(in_features=filter_length, out_features=filter_length),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        s = self.avpool(x)
        s = s.view(x.shape[0], -1)
        s = self.model(s).unsqueeze(-1)
        return x * s + s


class ResBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super(ResBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.BatchNorm1d(num_features=in_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm1d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same"),
            nn.MaxPool1d(kernel_size=3),
            FMS(filter_length=out_channels)  
        )
    def forward(self, x):
        return self.blocks(x)



class RawNet2(nn.Module):
    def __init__(self):
        super(RawNet2, self).__init__()
        self.sinc_filters = SincConv_fast(in_channels=1, out_channels=128, kernel_size=129)
        self.maxpool = nn.MaxPool1d(3)
        
        self.resblocks1 = nn.Sequential(
            ResBlock(in_channels=128, out_channels=128),
            ResBlock(in_channels=128, out_channels=128),
        )
        
        self.resblocks2 = nn.Sequential(
            ResBlock(in_channels=128, out_channels=512),
            ResBlock(in_channels=512, out_channels=512),
            ResBlock(in_channels=512, out_channels=512),
            ResBlock(in_channels=512, out_channels=512)
        )
        
        self.pre_gru = nn.Sequential(
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(),
        )
        
        self.gru = nn.GRU(input_size=512, hidden_size=1024, num_layers=3, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2)
        )
        
    def forward(self, audio, **batch):
        x = audio.unsqueeze(1) #(B, 1, 64000)
        print("Input:", x.shape)
        x = self.sinc_filters(x) #(B, 128, L)
        x = torch.abs(x)
        x = self.maxpool(x) #(B, 128, 21290)
        print("After sinc:", x.shape)
        x = self.resblocks1(x) #(B, 128, 2365)
        print("After resblock1:", x.shape)
        x = self.resblocks2(x) #(B, 512, 29)
        print("After resblock2:", x.shape)
        x = self.pre_gru(x) #(B, 512, 29)
        print("After pre_gru:", x.shape)
        x = x.transpose(-2, -1)
        x, _ = self.gru(x) #(B, 29, 1024)
        x = x[:, -1, :] #(B, 1024)
        print("After gru:", x.shape)
        x = self.fc(x) #(B, 2)
        print("After fc:", x.shape)
        
        return {"prediction": x}
        
        