import string

import torch
import numpy as np
from torch.utils.data import Dataset
from tslearn.clustering import KShape
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics.cluster import adjusted_rand_score


class TemporalSequence:


    def __init__(self, file_path, idx):
        self.features = np.load(file_path)
        self.features = self.features.reshape(self.features.shape[0], 2340)
        self.idx = idx

    def get_length(self):
        return self.features.shape[0]


def split_temporal_sequence(sequence, indices) -> list:
    """ According to the indices, split the temporal sequence to multiple sub-sequences.

    """
    assert all(indices[i] < indices[i + 1] for i in range(len(indices) - 1)), 'The index sequence must be increasing.'
    assert indices[0] != 0 or indices[-1] != sequence.shape[0], 'The temporal sequence is not totally split'

    split_sequences = [sequence[:, split_indices[i]:split_indices[i + 1]] for i in range(len(split_indices) - 1) ]
    return split_sequences

def brutally_split(totlen):
    result = [0 for _ in range(totlen)]
    for i in range(26):
        start_index = i * (totlen // 26)  # 起始索引
        end_index = (i + 1) * (totlen // 26)  # 结束索引（不包含）
        for k in range(start_index, end_index):
            result[k] = i  # 赋值为字母的 ASCII 码
        if i == 25:
            for k in range(end_index, totlen):
                result[k] = i
    return result

alphabet_sequence_indices = [1, 2, 7, 8]
def get_label(file_path, idx):
    temporal_sequence = TemporalSequence(file_path, idx)
    if idx in alphabet_sequence_indices:
        return brutally_split(temporal_sequence.get_length())
    return None


if  __name__ == '__main__':
    # 假设X是一个形状为(n_samples, n_timestamps, n_dimensions)的numpy数组
    # n_samples表示样本数量，n_timestamps表示时间步数量，n_dimensions表示每个时间点的特征数量
    X = random_walks(n_ts=50, sz=32, d=3)
    X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X)
    ks = KShape(n_clusters=3, n_init=1, random_state=0).fit(X)
    labels = ks.fit_predict(X)
    print(labels)