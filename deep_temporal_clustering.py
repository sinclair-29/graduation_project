import torch
import torch.nn as nn


class SequenceAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SequenceAutoencoder, self).__init__()

        # 1D CNN layer
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # BiLSTM layer
        self.bilstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        # Decoder
        self.decoder = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        # x: input sequence (batch_size, seq_length, input_size)

        # 1D CNN
        x = x.permute(0, 2, 1)  # Reshape for Conv1d
        x = self.cnn(x)

        # BiLSTM
        x = x.permute(0, 2, 1)  # Reshape for LSTM
        _, (h, _) = self.bilstm(x)
        hidden = torch.cat((h[-2], h[-1]), dim=1)  # Concatenate the hidden states of both directions

        # Decoder
        output = self.decoder(hidden)

        return output

# 创建模型实例
input_size = n  # 输入大小为 n
hidden_size = 64  # 隐藏层大小
num_layers = 2  # BiLSTM 层数
model = SequenceAutoencoder(input_size, hidden_size, num_layers)

# 定义均方误差损失函数
loss_fn = nn.MSELoss()

# 将输入数据转换为张量
input_data = torch.tensor(input_data)  # 输入数据的形状应为 (batch_size, seq_length, input_size)

# 前向传播
output_data = model(input_data)

# 计算损失
loss = loss_fn(output_data, input_data)