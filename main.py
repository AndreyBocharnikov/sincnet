import yaml
import time
import typing as tp 
import os

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
import wandb # sudo apt install libpython3.7-dev python3.7 -m pip install wandb
import numpy as np

from datasets.timit import TimitTrain, TimitVal
from model import SincNet
from utils import NestedNamespace


def compute_accuracy(logits: torch.Tensor, labels: tp.Union[torch.Tensor, int]) -> float:
    return torch.mean((torch.argmax(logits, dim=1) == labels).float()).item()


def main(params: NestedNamespace):
    chunk_len = int(params.sample_rate * params.chunk_len_ratio)
    chunk_shift = int(params.sample_rate * params.chunk_shift_ratio)
    dataset_train = TimitTrain(params.data.timit.path, chunk_len=chunk_len)
    dataset_val = TimitVal(params.data.timit.path, chunk_len, chunk_shift)
    dataloader = DataLoader(dataset_train, batch_size=params.batch_size, shuffle=True, drop_last=True, num_workers=2)

    sinc_net = SincNet(chunk_len, params.data.timit.n_classes)
    sinc_net = sinc_net.to(params.device)
    optim = torch.optim.RMSprop(sinc_net.parameters(), lr=params.lr, alpha=0.95, eps=1e-8)
    if params.model.pretrain is not None:
      checkpoint = torch.load(params.model.pretrain, map_location=torch.device(params.device))
      sinc_net.load_state_dict(checkpoint['model_state_dict'])
      optim.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = nn.CrossEntropyLoss()

    for i in range(params.n_epochs):
        accuracy, losses = [], []
        sinc_net.train()
        for j, batch in enumerate(dataloader):
            optim.zero_grad()
            chunks, labels = batch
            chunks, labels = chunks.to(params.device), labels.to(params.device)
            logits = sinc_net(chunks)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
        
            if i % params.verbose_every == 0:
                losses.append(loss.item())
                accuracy.append(compute_accuracy(logits, labels))

        if i % params.verbose_every == 0:
            sinc_net.eval()
            with torch.no_grad():
                chunks_accuracy = []
                losses_test = []
                wavs_accuracy = 0
                for chunks, label, n_chunks in dataset_val:
                    chunks = chunks.to(params.device)
                    logits = sinc_net(chunks)
                    loss = criterion(logits, torch.Tensor([label] * n_chunks).long().to(params.device))
                    
                    losses_test.append(loss.item())
                    chunks_accuracy.append(compute_accuracy(logits, label))
                    wavs_accuracy += (torch.argmax(logits.sum(dim=0)) == label).item()

                wandb.log({'train accuracy': np.mean(accuracy), 'train loss': np.mean(losses), 
                'test loss': np.mean(losses_test), 'test chunk accuracy': np.mean(chunks_accuracy), 
                'test wav accuracy': wavs_accuracy / len(dataset_val)})
                torch.save({'model_state_dict': sinc_net.state_dict(), 'optimizer_state_dict': optim.state_dict()},
                os.path.join(params.save_path, params.model.type + str(i) + '.pt'))


if __name__ == "__main__":
    with open('cfg.yaml') as config:
        params = yaml.load(config, Loader=yaml.FullLoader)
        params = NestedNamespace(params)
    if params.model.type not in ['cnn', 'sinc']:
      raise ValueError("Only two models are supported, use cnn or sinc.")
    wandb.init(project='SincNet', config={'model type': params.model.type})
    main(params)

# cd dr8 && for i in $( ls | grep [A-Z] ); do mv -i $i `echo $i | tr 'A-Z' 'a-z'`; done && cd ..
# for i in $( find dr8 -type f ); do mv -i $i `echo $i | tr 'A-Z' 'a-z'`; done
