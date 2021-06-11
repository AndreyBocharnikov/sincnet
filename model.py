import typing as tp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class SincConv(nn.Module):
    def init_params_mel(self, out_channels):
        def to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        return to_hz(np.linspace(to_mel(self.min_hz),
                                 to_mel(self.sample_rate / 2 - self.min_hz - self.min_band),
                                 out_channels + 1))

    def __init__(self, in_channels, out_channels, kernel_size, padding, sample_rate=16000, min_hz=50, min_band=50):
        super().__init__()
        self.sample_rate, self.min_hz, self.min_band = sample_rate, min_hz, min_band
        self.padding = padding
        hz = self.init_params_mel(out_channels)
        self.hz_left = nn.Parameter(torch.unsqueeze(torch.Tensor(hz[:-1]), dim=1))
        self.hz_band = nn.Parameter(torch.unsqueeze(torch.Tensor(np.diff(hz)), dim=1))
        self.window = torch.hann_window(kernel_size)[:kernel_size // 2]
        self.n = 2 * np.pi * torch.unsqueeze(torch.arange(-(kernel_size // 2), 0), dim=0) / sample_rate

    def forward(self, wav):
        self.window, self.n = self.window.to(wav.device), self.n.to(wav.device)

        low = self.min_hz + torch.abs(self.hz_left)
        high = low + self.min_band + self.hz_band
        band = (high - low)[:, 0]

        f_low = torch.matmul(low, self.n)
        f_high = torch.matmul(high, self.n)

        band_pass_left = 2 * (torch.sin(f_high) - torch.sin(f_low)) / self.n * self.window
        band_pass_center = 2 * torch.unsqueeze(band, dim=1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat((band_pass_left, band_pass_center, band_pass_right), dim=1) / band_pass_center

        filters = torch.unsqueeze(band_pass, dim=1)
        return F.conv1d(wav, filters, padding=self.padding)


class CNNBlock(nn.Module):
    def __init__(self, seq_len: int, conv_type: tp.Union[tp.Type[nn.Conv1d], tp.Type[SincConv]],
                 in_channels: int, out_channels: int, kernel_size: int, pool_size: int = 3, dropout_p: float = 0.0):
        super().__init__()
        conv_block = conv_type(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        pooling = nn.MaxPool1d(pool_size)
        ln = nn.LayerNorm(seq_len // pool_size)
        lrelu = nn.LeakyReLU()
        dropout = nn.Dropout(p=dropout_p)

        self.net = nn.Sequential(conv_block, pooling, ln, lrelu, dropout)

    def forward(self, x):
        return self.net(x)


class MLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_p: float = 0.0):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        bn = nn.BatchNorm1d(out_features, momentum=0.05)
        lrelu = nn.LeakyReLU()
        dropout = nn.Dropout(p=dropout_p)

        self.net = nn.Sequential(linear, bn, lrelu, dropout)

    def forward(self, x):
        return self.net(x)


class SincNet(nn.Module):
    def __init__(self, chunk_len, n_classes, conv_type: str):
        super().__init__()
        if conv_type not in ['cnn', 'sinc']:
            raise ValueError("Only two types of first convolution are supported, cnn and sinc.")
        if conv_type == 'cnn':
            first_block = nn.Conv1d
        else:
            first_block = SincConv
        ln1 = nn.LayerNorm(chunk_len)
        cnn_blocks = nn.Sequential(CNNBlock(chunk_len, first_block, 1, 80, 251),
                                   CNNBlock(chunk_len // 3, nn.Conv1d, 80, 60, 5),
                                   CNNBlock(chunk_len // 9, nn.Conv1d, 60, 60, 5))
        flatten = nn.Flatten(start_dim=1)
        ln2 = nn.LayerNorm(chunk_len // 27 * 60)
        mlp_blocks = nn.Sequential(MLPBlock(chunk_len // 27 * 60, 2048),
                                   MLPBlock(2048, 2048),
                                   MLPBlock(2048, 2048))
        self.backbone = nn.Sequential(ln1, cnn_blocks, flatten, ln2, mlp_blocks)
        self.classification_head = nn.Linear(2048, n_classes)

    def forward(self, wavs):
        return self.classification_head(self.backbone(wavs))

    def compute_d_vectors(self, wavs):
        return self.backbone(wavs)


if __name__ == "__main__":
    sincnet = SincNet(3200, 462, 'sinc')
    #print(summary(sincnet, (1, 3200)))
    model_parameters = filter(lambda p: p.requires_grad, sincnet.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

# 22906880 vs 45358