from bitnet import *
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# MNISTデータセットのダウンロードと前処理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# ニューラルネットワークの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 =BitLinearOriginal(784, 128)
        self.fc2 =BitLinearOriginal(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# オプティマイザのパラメータを記録するためのリストを初期化
param_history = []

def calculate_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy

# 学習ループ
for epoch in range(50):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # オプティマイザのパラメータを記録
        param_history.append([p.clone().detach().numpy() for p in model.parameters()])

        optimizer.step()
        acc = calculate_accuracy(output, target)
        print(f"loss:{loss}  acc:{acc}")
        if acc>0.99:
          break
