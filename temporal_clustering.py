import string
import random

import torch
import numpy as np
from torch.utils.data import Dataset
from tslearn.clustering import KShape
from tslearn.clustering import TimeSeriesKMeans
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

    def get_sequence(self):
        return self.features

    def get_split_number(self):
        if self.idx == 3:
            return 130
        if self.idx in [4, 5, 9]:
            return 142
        if self.idx == 6:
            return 800

    def get_features(self):
        return self.features

def pad_array(arr, target_shape):
    x, y = arr.shape
    z = target_shape
    padding_rows = z - x
    if padding_rows == 0:
        return arr
    padding = np.zeros((padding_rows, y), dtype=arr.dtype)
    padded_arr = np.vstack((arr, padding))
    return padded_arr

def split_temporal_sequence(sequence, indices) -> list:
    """ According to the indices, split the temporal sequence to multiple sub-sequences.

    """
    indices = [0] + indices + [sequence.get_length()]
    assert all(indices[i] < indices[i + 1] for i in range(len(indices) - 1)), 'The index sequence must be increasing.'

    split_sequences = [sequence.get_features()[indices[i]:indices[i + 1], :] for i in range(len(indices) - 1) ]
    max_length = 0
    for i in range(len(indices) - 1):
        max_length = max_length if max_length > indices[i + 1] - indices[i] else indices[i + 1] - indices[i]
    return split_sequences, max_length

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

def check_adjust(current_idx, remained_num, reminder, split_number):
    if remained_num <= 0:
        return False
    float_adjust_point = (1.0 * split_number / reminder) * (reminder - remained_num + 1)
    if current_idx >=  float_adjust_point:
        return True
    else:
        return False

def slightly_perturbation(x):
    return random.choice([-1, 0, 1]) + x


def initialize_population(population_size, sequence_length, split_number):
    # generate the base split
    base_split = []
    base_size = sequence_length // split_number
    remainder = sequence_length % split_number
    adjusted_segment_num = remainder
    current_position = 0
    for i in range(1, split_number):
        current_position += base_size
        if check_adjust(i, adjusted_segment_num, remainder, split_number):
            current_position += 1
            adjusted_segment_num -= 1
        base_split.append(current_position)

    population = []
    for _ in range(population_size):
        chromosome = [slightly_perturbation(x) for x in base_split]
        population.append(chromosome)
    return population

def fitness_function(true_labels, solution, temporal_sequence):
    #print(solution)

    split_sequences, max_length = split_temporal_sequence(temporal_sequence, solution)
    padded_split_sequences = [pad_array(_, max_length) for _ in split_sequences]

    clustering_sequence = np.array(padded_split_sequences)
    ks = TimeSeriesKMeans(n_clusters=26, n_init=2, random_state=0).fit(clustering_sequence)
    predicted_labels = ks.fit_predict(clustering_sequence)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    return ari, predicted_labels

def get_label_list(char_sequence):
    result = [ord(char) - ord('a') for char in char_sequence]
    return result

def get_true_label(idx):
    if idx == 3:
        char_sequence = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxuzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
    elif idx in [4, 5, 9]:
        char_sequence = "privacyiscriticalforensuringthesecurityofcomputersystemsandtheprivacyofhumanusersaswhatbeingtypescouldbepasswordsorprivacysensitiveinformation"
    else:
        char_sequence = "privacyiscriticalforenuringthesecurityofcomputersystemsandtheprivacyofhumanusersaswhaybeingtypescouldbepasswordsorprivacysensitivesinformationtheresearchcommunityhassutdiedvariouswaystorecognizekeystrokeswhichcanbeclassifiedintothreecategoroesacousticemissionbasedapproacheselectromagneticemmisionbasedapproachesandvisionbasedapprachesacousticemmissionabasedapproachesrecognizekeystrokesbasedontethiertheobservationthattypingsoundsortheobservationthattheacousticemanationfromdifferentkeysarribeaydirrerenttimeasthekeysarelocatedatdifferentplacesinakeyboardelectromagneticemmissionbasedapproachesrecognizekeystrokesbasedontheobsrvationthattheelecyromagneticemanationsfromtheelectrivalvircuitunderneathdifferentkeysinakeyboardaredifferentvisionbasedapproachesrecognizekeystrokeusingvisiontechnologies"
    return get_label_list(char_sequence)


def generate_label_list(indices, labels, length):
    indices = [0] + indices + [length]
    result = [0 for _ in range(length)]
    for i in range(len(labels)):
        for j in range(indices[i], indices[i + 1]):
            result[j] = labels[i]
    return result

alphabet_sequence_indices = [1, 2, 7, 8]
def get_label(file_path, idx):  
    temporal_sequence = TemporalSequence(file_path, idx)
    if idx in alphabet_sequence_indices:
        return brutally_split(temporal_sequence.get_length())
    else:
        split_number = temporal_sequence.get_split_number()
        sequence_length = temporal_sequence.get_length()
        population_size = (sequence_length - 1) * 3
        population = initialize_population(population_size, sequence_length, split_number)
        true_label = get_true_label(idx)

        best_solution = None
        best_score = -2
        for i in range(population_size):
            fitness, label = fitness_function(true_label, population[i], temporal_sequence)
            if fitness > best_score:
                best_solution = population[i]
                best_score = fitness
    return generate_label_list(best_solution, true_label, temporal_sequence.get_length())





if  __name__ == '__main__':
    # 假设X是一个形状为(n_samples, n_timestamps, n_dimensions)的numpy数组
    # n_samples表示样本数量，n_timestamps表示时间步数量，n_dimensions表示每个时间点的特征数量
    test_path = "bfa/bfa_4.npy"
    print(get_label(test_path, 4))
