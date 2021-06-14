import yaml
import typing as tp
import os

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
import wandb  # sudo apt install libpython3.7-dev python3.7 -m pip install wandb
import numpy as np

from datasets.timit import TimitTrain, TimitEval
from model.model import SincNet
from utils import NestedNamespace, compute_chunk_info


def compute_accuracy(logits: torch.Tensor, labels: tp.Union[torch.Tensor, int]) -> float:
    return torch.mean((torch.argmax(logits, dim=1) == labels).float()).item()


def main(params: NestedNamespace):
    chunk_len, chunk_shift = compute_chunk_info(params)
    dataset_train = TimitTrain(params.data.timit.path, chunk_len=chunk_len)
    dataset_evaluation = TimitEval(params.data.timit.path, chunk_len, chunk_shift, 'test.scp')
    dataloader = DataLoader(dataset_train, batch_size=params.batch_size, shuffle=True, drop_last=True, num_workers=2)

    sinc_net = SincNet(chunk_len, params.data.timit.n_classes, params.model.type)
    sinc_net = sinc_net.to(params.device)
    optim = torch.optim.RMSprop(sinc_net.parameters(), lr=params.lr, alpha=0.95, eps=1e-8)
    prev_epoch = 0
    if params.model.pretrain is not None:
        checkpoint = torch.load(params.model.pretrain, map_location=torch.device(params.device))
        sinc_net.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epoch = checkpoint['epoch'] + 1
    criterion = nn.CrossEntropyLoss()

    for i in range(prev_epoch, prev_epoch + params.n_epochs):
        accuracy, losses = [], []
        sinc_net.train()
        for j, batch in enumerate(dataloader):
            print(j)
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
                chunks_accuracy, losses_test = [], []
                wavs_accuracy = 0
                for chunks, label, n_chunks in dataset_evaluation:
                    chunks = chunks.to(params.device)
                    logits = sinc_net(chunks)
                    loss = criterion(logits, torch.Tensor([label] * n_chunks).long().to(params.device))

                    losses_test.append(loss.item())
                    chunks_accuracy.append(compute_accuracy(logits, label))
                    wavs_accuracy += (torch.argmax(logits.sum(dim=0)) == label).item()

                if params.use_wandb:
                    wandb.log({'train accuracy': np.mean(accuracy), 'train loss': np.mean(losses),
                               'test loss': np.mean(losses_test), 'test chunk accuracy': np.mean(chunks_accuracy),
                               'test wav accuracy': wavs_accuracy / len(dataset_evaluation), 'epoch': i})
                else:
                    print(f'epoch {i}\ntrain accuracy {np.mean(accuracy)}\ntrain loss {np.mean(losses)} \n'
                          f'test loss {np.mean(chunks_accuracy)}\ntest chunk accuracy {np.mean(chunks_accuracy)}\n'
                          f'test wav accuracy {wavs_accuracy / len(dataset_evaluation)}')
                torch.save(
                    {'model_state_dict': sinc_net.state_dict(), 'optimizer_state_dict': optim.state_dict(), 'epoch': i},
                    os.path.join(params.save_path, params.model.type + str(i) + '.pt'))


if __name__ == "__main__":
    with open('configs/cfg.yaml') as config:
        params = yaml.load(config, Loader=yaml.FullLoader)
        params = NestedNamespace(params)
    if params.model.type not in ['cnn', 'sinc']:
        raise ValueError("Only two models are supported, use cnn or sinc.")
    if params.use_wandb:
        id = wandb.util.generate_id()
        print("id", id)
        wandb.init(project='SincNet', id=id, resume="allow", config={'model type': params.model.type})
    main(params)
