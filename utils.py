import typing as tp
from types import SimpleNamespace


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)


def compute_chunk_info(params: NestedNamespace) -> tp.Tuple[int, int]:
    chunk_len = int(params.sample_rate * params.chunk_len_ratio)
    chunk_shift = int(params.sample_rate * params.chunk_shift_ratio)
    return chunk_len, chunk_shift
