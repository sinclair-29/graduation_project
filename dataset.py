import torch
import numpy as np
from torch.utils.data import Dataset

from temporal_clustering import get_label

class AngleDataset(Dataset):


    def __init__(self, file_path, idx, type):
        self.features = np.load(file_path)
        self.features = torch.from_numpy(self.features).float()
        if type == 'static':
            self.labels = torch.full((len(self.features), ), idx)
        elif type == 'dynamic':
            self.labels = get_label(file_path, idx)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature = torch.unsqueeze(feature, 0)  # 在第 0 维度添加一个维度
        label = self.labels[idx]
        return feature, label