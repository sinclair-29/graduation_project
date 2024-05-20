import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from torcheval.metrics.functional import multiclass_f1_score

from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import logging
import string
import torch.nn.functional as F
import time
from PIL import Image

num_frame = [0, 186, 172, 794, 708, 634, 3927, 160, 229, 694]
num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
paths = []
for num in num_list:
    path = "bfa/bfa_" + str(num) + ".npy"
    paths.append(path)
datasets = []



def get_label(idx, totlen):
    # 如果是26键入字母
    result = [0 for _ in range(totlen)]
    if idx in [1, 2, 7, 8]:
        for i in range(26):
            start_index = i * (totlen // 26)  # 起始索引
            end_index = (i + 1) * (totlen // 26)  # 结束索引（不包含）
            for k in range(start_index, end_index):
                result[k] = i  # 赋值为字母的 ASCII 码
            if i == 25:
                for k in range(end_index, totlen):
                    result[k] = i
        #print(result)
        return result
    if idx == 3:
        for i in range(26 * 5):
            start_index = i * (totlen // 130)  # 起始索引
            end_index = (i + 1) * (totlen // 130)  # 结束索引（不包含）
            for k in range(start_index, end_index):
                result[k] = i % 26
            if i == 26 * 5:
                for k in range(end_index, totlen):
                    result[k] = 25
        return result
    if idx in [4, 5, 9]:
        str = "privacyiscriticalforensuringthesecurityofcomputersystemsandtheprivacyofhumanusersaswhatbeingtypescouldbepasswordsorprivacysensitiveinformation"
        strlen = len(str)
        for i in range(strlen):
            start_index = i * (totlen // strlen)  # 起始索引
            end_index = (i + 1) * (totlen // strlen)  # 结束索引（不包含）
            for k in range(start_index, end_index):
                result[k] = ord(str[i]) - ord('a')
            if i == strlen:
                for k in range(end_index, totlen):
                    result[k] = ord(str[i]) - ord('a')
        return result
    if idx == 6:
        str = "privacyiscriticalforenuringthesecurityofcomputersystemsandtheprivacyofhumanusersaswhaybeingtypescouldbepasswordsorprivacysensitivesinformationtheresearchcommunityhassutdiedvariouswaystorecognizekeystrokeswhichcanbeclassifiedintothreecategoroesacousticemissionbasedapproacheselectromagneticemmisionbasedapproachesandvisionbasedapprachesacousticemmissionabasedapproachesrecognizekeystrokesbasedontethiertheobservationthattypingsoundsortheobservationthattheacousticemanationfromdifferentkeysarribeaydirrerenttimeasthekeysarelocatedatdifferentplacesinakeyboardelectromagneticemmissionbasedapproachesrecognizekeystrokesbasedontheobsrvationthattheelecyromagneticemanationsfromtheelectrivalvircuitunderneathdifferentkeysinakeyboardaredifferentvisionbasedapproachesrecognizekeystrokeusingvisiontechnologies"
        strlen = len(str)
        for i in range(strlen):
            start_index = i * (totlen // strlen)  # 起始索引
            end_index = (i + 1) * (totlen // strlen)  # 结束索引（不包含）
            for k in range(start_index, end_index):
                result[k] = ord(str[i]) - ord('a')
            #print(ord(str[i]) - ord('a'))
            if i == strlen:
                for k in range(end_index, totlen):
                    result[k] = ord(str[i]) - ord('a')
        #print(result)
        return result





class angleDataset(Dataset):
    def __init__(self, file_path, idx):
        self.features = np.load(file_path)
        self.features = torch.from_numpy(self.features).float()
        #self.labels = torch.full((len(self.features), ), idx)
        self.labels = get_label(idx, len(self.features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature = torch.unsqueeze(feature, 0)  # 在第 0 维度添加一个维度
        label = self.labels[idx]
        return feature, label

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layer = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #第三层
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.maxpool_layer = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.fc_layer1 = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(128 * 58 * 2, 64),
            #nn.Dropout(0.5)
        )
        self.fc_layer2 = nn.Linear(64, 26)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.maxpool_layer(x)
        #print(x.shape)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        return x

def fit(epoch, model, data_loader, phase="training",
        device=torch.device("cpu"), log_interval=10000, criterion=nn.CrossEntropyLoss()):
    if phase == "training":
        model.train()
    if phase == "validation":
        model.eval()
    running_loss = 0.0
    running_correct = 0
    total_acc, total_count, total_loss = 0, 0, 0
    predicted_labels = []
    true_labels = []
    for idx, (data, label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device)
        if phase == "training":
            optimizer.zero_grad()
        output = model(data)
        predicted_label = output.data.max(dim=1, keepdim=True)[1]
        #print(output)
        #print(label)
        #print(predicted_label)
        loss = criterion(output, label)
        running_loss += loss.item()
        running_correct += predicted_label.eq(label.data.view_as(predicted_label)).cpu().sum()

        tmp_batch_size = predicted_label.shape[0]
        for i in range(tmp_batch_size):
            predicted_labels.append(predicted_label[i][0].item())
            true_labels.append(label[i].item())

        if phase == "training":
            loss.backward()
            optimizer.step()
            total_acc += predicted_label.eq(label.data.view_as(predicted_label)).cpu().sum()
            total_count += data.shape[0]
            total_loss += loss.item()
            if idx > 0 and idx % log_interval == 0:
                logging.info(f'| epoch {epoch} | {idx:2d}/{len(data_loader)} batches '
                             f'| accuracy {total_acc / total_count:.4f}'
                             f'| loss {loss.item():.4f}')
                total_acc, total_count, total_loss = 0, 0, 0
    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)
    f1_micro = f1_score(true_labels, predicted_labels, average='macro')
    if phase == "validation":
        logging.info(f'loss : {loss}')
    else:
        if phase == "testing":
            logging.info(f'Macro-average F1 Score: {f1_micro}')
    return loss, accuracy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%a %d %b %Y %H:%M:%S" # 星期, 日, 月, 年份, 时间
)

# prepare the dataset
"""
alphabet_list = list(string.ascii_lowercase)

paths = []
for alpha in alphabet_list:
    path = "bfa/bfa_" + alpha + ".npy"
    paths.append(path)
datasets = []
"""

for i in range(1, 10):
    if i not in [1, 2, 7, 8]:
        continue
    datasets.append(angleDataset(paths[i - 1], i))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
combined_dataset = ConcatDataset(datasets)
train_ratio = 0.6  # 训练集比例
val_ratio = 0.2 # 校验集比例
test_ratio = 0.2  # 测试集比例
batch_size = 64
learning_rate = 0.001

"""
feature, label = combined_dataset[0]
feature = np.squeeze(feature, axis=0)

print(feature.shape)
# 将feature转换为NumPy数
feature_array = np.array(feature, dtype=np.uint8)

# 创建灰度图像
gray_image = Image.fromarray(feature_array, mode='L')

# 保存灰度图像
gray_image.save('feature_gray.png')

"""
# 计算划分的样本
train_size = int(train_ratio * len(combined_dataset))
val_size = int(val_ratio * len(combined_dataset))
test_size = len(combined_dataset) - train_size - val_size

# 使用 random_split 进行划分
train_dataset, val_dataset, test_dataset = random_split(combined_dataset, [train_size, val_size, test_size])

criterion = torch.nn.CrossEntropyLoss()
model = ConvNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
valid_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

best_loss = 1e9
patient = 0
for epoch in range(1, 400 + 1):
    fit(epoch, model, train_dataloader, phase="training", device=device)
    training_loss, training_tmp = fit(epoch, model, valid_dataloader, phase="validation", device=device)
    if training_loss > 0.05 and best_loss == 1e9:
        continue
    if training_loss < best_loss:
        best_loss = training_loss
        patient = 0
    else:
        patient += 1
        if patient >= 25:
            break

loss, accuracy = fit(0, model, test_dataloader, phase="testing", device=device)
logging.info('训练完成')
logging.info(f'在测试集上的loss: {loss}')
logging.info(f'在测试集上的accuracy: {accuracy}' ) 
