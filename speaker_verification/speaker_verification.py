from argparse import ArgumentParser, Namespace
from collections import defaultdict
import typing as tp

from scipy.spatial import distance
import numpy as np
import torch
import yaml

from datasets.timit import TimitEval
from model import SincNet
from utils import NestedNamespace, compute_chunk_info, get_params, load_model


def compute_cosine_dists(model, chunks, label, speakers: tp.List[np.ndarray]):
    d_vectors_chunk = model.compute_d_vectors(chunks)
    cur_d_vector = (d_vectors_chunk / d_vectors_chunk.norm(p=2, dim=1, keepdim=True)).mean(dim=0)
    distances = []
    for speaker in speakers:
        distances.append(distance.cosine(cur_d_vector, speaker))
    return distances


def compute_softmax_probs(model, chunks, label, speakers: tp.List[np.ndarray]):
    logits = model.classification_head(torch.Tensor(speakers))[:, label]
    return torch.softmax(logits, dim=0).numpy().tolist()


def compute_eer(logits: tp.List[tp.Tuple[float, bool]]):
    threshold = sorted(logits)
    num_negatives, num_positives = len(logits) - len(logits) // 11, len(logits) // 11
    min_diff, result = 1, None
    fp, fn = 1, 0
    for _, class_ in threshold:
        if class_:
            fn += 1 / num_positives
        else:
            fp -= 1 / num_negatives
        if min_diff > abs(fn - fp):
            min_diff = abs(fn - fp)
            result = (fn + fp) / 2
    return result


def main(params: NestedNamespace, args: Namespace, setup: tp.Union[compute_cosine_dists, compute_softmax_probs]):
    chunk_len, chunk_shift = compute_chunk_info(params)
    sinc_net = load_model(params, args, chunk_len)
    d_vectors = np.load(args.d_vectors, allow_pickle=True).item()

    evaluation_train = TimitEval(params.data.timit.path, chunk_len, chunk_shift, 'train.scp')
    logits = []
    with torch.no_grad():
        for chunks, label, _ in evaluation_train:
            while True:
                imposters_ids = np.random.choice(params.data.timit.n_classes, 10)
                if label not in imposters_ids:
                    break
            imposters = [d_vectors.get(imposter) for imposter in imposters_ids]
            speakers = imposters + [d_vectors.get(label)]
            values = setup(sinc_net, chunks, label, speakers)
            logits += values
    classes = ([False] * 10 + [True]) * len(evaluation_train)
    logits = list(zip(logits, classes))
    eer = compute_eer(logits)
    print(eer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('pretrained_model')
    parser.add_argument('d_vectors')
    parser.add_argument('setup',
                        help='whether to use cosine distance or softmax posterior score, should be cos or softmax')
    args = parser.parse_args()
    if args.setup not in ['cos', 'softmax']:
        raise ValueError('setup argument should be cos or softmax')
    if args.setup == 'cos':
        setup = compute_cosine_dists
    else:
        setup = compute_softmax_probs
    params = get_params('cfg.yaml')

    main(params, args, setup)
