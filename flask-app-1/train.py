import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class PostureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
def train_model(epochs = 20):
    #read data
    normal_data = []
    abnormal_data = []  
    for file in os.listdir('data/normal'):
            normal_data.append(np.load(os.path.join('data/normal', file)))
    for file in os.listdir('data/abnormal'):
            abnormal_data.append(np.load(os.path.join('data/abnormal', file)))
    # Create labels
    normal_labels = np.ones(len(normal_data))
    abnormal_labels = np.zeros(len(abnormal_data))
    #create dataset
    features = np.concatenate([normal_data, abnormal_data], axis=0)
    labels = np.concatenate([normal_labels, abnormal_labels], axis=0)
    #to tensor
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    dataset = PostureDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 4. 定義簡單的二元分類模型
    class PostureClassifier(nn.Module):
        def __init__(self, input_dim):
            super(PostureClassifier, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # 輸出為 0-1 的機率
            )

        def forward(self, x):
            return self.fc(x)

    # 5. 初始化模型、損失函數和優化器
    input_dim = features.shape[1]  # 特徵數量
    model = PostureClassifier(input_dim)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. 訓練模型
    for epoch in range(epochs):
        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            batch_labels = batch_labels.float().squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


    with torch.no_grad():
            outputs = model(features).squeeze()
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == labels).sum() / len(labels)
            print(f"Accuracy: {accuracy:.4f}")

    # 儲存模型
    torch.save(model.state_dict(), "posture_classifier.pth")
    return accuracy, model