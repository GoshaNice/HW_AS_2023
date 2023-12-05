import torch.nn as nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, channels, kernel, dilations, neg_slope=0.1):
        super(ResBlock, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(dilations)):
            block = nn.Sequential(
                nn.LeakyReLU(negative_slope=neg_slope),
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel,
                    padding="same",
                    dilation=dilations[i],
                ),
                nn.LeakyReLU(negative_slope=neg_slope),
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel,
                    padding="same",
                    dilation=1,
                ),
            )
            self.blocks.append(block)

    def forward(self, x):
        output = 0
        for block in self.blocks:
            output = output + block(x)
        return output



class RawNet2(nn.Module):
    def __init__(self):
        self.sinc_filters = nn.Sequential(
            nn.Conv1d(1, 129, 128),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(129),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        return x
        
        