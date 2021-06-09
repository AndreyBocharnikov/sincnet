import typing as tp

import torch
import torch.nn as nn


class SincConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


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
    def __init__(self, wav_len, n_classes):
        super().__init__()
        ln1 = nn.LayerNorm(wav_len)
        cnn_blocks = nn.Sequential(CNNBlock(wav_len, nn.Conv1d, 1, 80, 251),
                                        CNNBlock(wav_len // 3, nn.Conv1d, 80, 60, 5),
                                        CNNBlock(wav_len // 9, nn.Conv1d, 60, 60, 5))
        flatten = nn.Flatten(start_dim=1)
        ln2 = nn.LayerNorm(wav_len // 27 * 60)
        mlp_blocks = nn.Sequential(MLPBlock(wav_len // 27 * 60, 2048),
                                        MLPBlock(2048, 2048),
                                        MLPBlock(2048, 2048))
        classification_head = nn.Linear(2048, n_classes)

        self.net = nn.Sequential(ln1, cnn_blocks, flatten, ln2, mlp_blocks, classification_head)

    def forward(self, wavs):
        return self.net(wavs)


if __name__ == "__main__":
    wav_len = 16000 // 5
    x = torch.rand((128, 1, wav_len))
    net = SincNet(wav_len, 10)
    print(net(x).shape)