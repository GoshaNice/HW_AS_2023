import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.model.SincConv import SincConv_fast


class FMS(nn.Module):
    def __init__(self, filter_length: int):
        super(FMS, self).__init__()
        self.avpool = nn.AdaptiveAvgPool1d(1)
        self.model = nn.Sequential(
            nn.Linear(in_features=filter_length, out_features=filter_length),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.avpool(x)
        s = s.view(x.shape[0], -1)
        s = self.model(s).unsqueeze(-1)
        return x * s + s


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, neg_slope: float = 0.3):
        super(ResBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(num_features=in_channels),
            nn.LeakyReLU(negative_slope=neg_slope),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=neg_slope),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
        )

        if in_channels != out_channels:
            self.resample = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )

        self.block2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3), FMS(filter_length=out_channels)
        )

    def forward(self, x):
        out = self.block1(x)
        if out.shape[1] != x.shape[1]:
            x = self.resample(x)
        out = out + x
        out = self.block2(out)
        return out


class RawNet2(nn.Module):
    def __init__(
        self,
        sinc_filter_length: int = 1024,
        sinc_channels: int = 128,
        min_low_hz: int = 0,
        min_band_hz: int = 0,
        resblock1_channels: int = 20,
        resblock2_channels: int = 128,
        gru_hidden: int = 1024,
        gru_num_layers: int = 3,
        fc_hidden: int = 1024,
        neg_slope: float = 0.3,
        use_abs: bool = True,
        s3: bool = False,
    ):
        super(RawNet2, self).__init__()
        self.sinc_filters = SincConv_fast(
            in_channels=1,
            out_channels=sinc_channels,
            kernel_size=sinc_filter_length,
            min_low_hz=min_low_hz,
            min_band_hz=min_band_hz,
            s3=s3,
        )
        self.use_abs = use_abs
        self.maxpool = nn.MaxPool1d(3)

        self.resblocks1 = nn.Sequential(
            ResBlock(
                in_channels=sinc_channels,
                out_channels=resblock1_channels,
                neg_slope=neg_slope,
            ),
            ResBlock(
                in_channels=resblock1_channels,
                out_channels=resblock1_channels,
                neg_slope=neg_slope,
            ),
        )

        self.resblocks2 = nn.Sequential(
            ResBlock(
                in_channels=resblock1_channels,
                out_channels=resblock2_channels,
                neg_slope=neg_slope,
            ),
            ResBlock(
                in_channels=resblock2_channels,
                out_channels=resblock2_channels,
                neg_slope=neg_slope,
            ),
            ResBlock(
                in_channels=resblock2_channels,
                out_channels=resblock2_channels,
                neg_slope=neg_slope,
            ),
            ResBlock(
                in_channels=resblock2_channels,
                out_channels=resblock2_channels,
                neg_slope=neg_slope,
            ),
        )

        self.pre_gru = nn.Sequential(
            nn.BatchNorm1d(num_features=resblock2_channels),
            nn.LeakyReLU(neg_slope),
        )

        self.gru = nn.GRU(
            input_size=resblock2_channels,
            hidden_size=gru_hidden,
            num_layers=gru_num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=gru_hidden, out_features=fc_hidden),  # to embeddings
            nn.Linear(in_features=fc_hidden, out_features=2),  # output
        )

    def forward(self, audio, **batch):
        x = audio.unsqueeze(1)  # (B, 1, 64000)
        x = self.sinc_filters(x)  # (B, 1024, L)
        if self.use_abs:
            x = torch.abs(x)
        x = self.maxpool(x)  # (B, 1024, 21290)
        x = self.resblocks1(x)  # (B, 20, 2365)
        x = self.resblocks2(x)  # (B, 128, 29)
        x = self.pre_gru(x)  # (B, 128, 29)
        x = x.transpose(-2, -1)
        x, _ = self.gru(x)  # (B, 29, 1024)
        x = x[:, -1, :]  # (B, 1024)
        x = self.fc(x)  # (B, 2)
        if not self.training:
            x = F.softmax(x, dim=-1)

        return {"prediction": x}
