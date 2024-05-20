import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader

class angleDataset(Dataset):
    def __init__(self, file_path, label):
        self.data = np.load(file_path)
        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.full((len(self.data), ), label)
        print(self.data.shape)
        print(self.labels.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 121 * 5, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

obs_path = 'bfa/bfa_obs.npy'
noobs_path = 'bfa/bfa_noobs.npy'
obs_dataset = angleDataset(obs_path, "obs")
noobs_dataset = angleDataset(noobs_path, "no obs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
test_loader = DataLoader(obs_dataset, batch_size=64, shuffle=True)

for i in range(2):
    test_data, test_label = next(iter(test_loader))
    print(test_data)
    print(test_label)

def train(dataloader, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 1
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)


        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time

            logging.info(f'| epoch {epoch} | {idx}/{len(dataloader)} batches '
                            f'| accuracy {total_acc/total_count}')

            total_acc, total_count = 0, 0
            start_time = time.time()

