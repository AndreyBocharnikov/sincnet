import yaml
import time

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
import wandb # sudo apt install libpython3.7-dev python3.7 -m pip install wandb
import numpy as np

from datasets.timit import TimitTrain
from model import SincNet
from utils import NestedNamespace

with open('cfg.yaml') as config:
    params = yaml.load(config, Loader=yaml.FullLoader)
    params = NestedNamespace(params)
#wandb.init(project='SincNet')
#wandb.config = params

wav_len = int(params.sample_rate * params.chunk_len)
dataset = TimitTrain(params.data.timit.path, chunk_len=params.chunk_len, sr=params.sample_rate)
dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, drop_last=True, num_workers=2)

sinc_net = SincNet(wav_len, params.data.timit.n_classes)
sinc_net = sinc_net.to(params.device)
optim = torch.optim.RMSprop(sinc_net.parameters(), lr=params.lr, alpha=0.95, eps=1e-8)
criterion = nn.CrossEntropyLoss()

for i in range(params.n_epochs):
    accuracy, losses = [], []
    for j, batch in enumerate(dataloader):
        optim.zero_grad()
        chunks, labels = batch
        chunks, labels = chunks.to(params.device), labels.to(params.device)
        logits = sinc_net(chunks)
        loss = criterion(logits, labels)
        loss.backward()
        optim.step()
    
        losses.append(loss.item())
        accuracy.append(torch.mean((torch.argmax(logits, dim=1) == labels).float()).item())
    if i % 400 == 0:
      print("mean loss", np.mean(losses))
      print("mean accuracy", np.mean(accuracy))
      print()
    #wandb.log({'accuracy': np.mean(accuracy), 'loss': np.mean(losses)})


# cd dr8 && for i in $( ls | grep [A-Z] ); do mv -i $i `echo $i | tr 'A-Z' 'a-z'`; done && cd ..
# for i in $( find dr8 -type f ); do mv -i $i `echo $i | tr 'A-Z' 'a-z'`; done
