import torch.nn as nn


class ConvNet(nn.Module):


    def __init__(self, num_class = 26):
        super(ConvNet, self).__init__()
        self.conv_layer = nn.Sequential(
            #conv3-32
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # conv3-64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv3-128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 58 * 2, 64),
            nn.Linear(64, num_class)
        )

    def forward(self, x):
        feature = self.conv_layer(x)
        flattened_feature = feature.reshape(feature.size(0), -1)
        out = self.fc_layer(flattened_feature)
        return out