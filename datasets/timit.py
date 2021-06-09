import os

import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio


class TimitTrain(Dataset):
    def __init__(self, path, chunk_len, sr):
        self.sr = sr
        self.chunk_len = chunk_len
        self.path = path
        self.labels = np.load(os.path.join(path, 'labels.npy'), allow_pickle=True).item()
        with open(os.path.join(path, 'train.scp')) as f:
            self.train_samples = f.read().splitlines()

    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, idx):
        wav_path = self.train_samples[idx]
        label = self.labels[wav_path]
        wav, sr = torchaudio.load(os.path.join(self.path, wav_path))
        assert sr == self.sr
        n_sample_chunk = int(self.sr * self.chunk_len)
        if n_sample_chunk > len(wav[0]):
            print(wav_path)
            assert False
        start = np.random.randint(len(wav[0]) - n_sample_chunk)
        chunk = wav[0, start:start + n_sample_chunk]
        volume_gain = 0.4 * torch.rand(1) + 0.8
        return torch.unsqueeze(chunk, 0) * volume_gain, label
