import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dữ liệu
train_pairs = np.load('train_pairs.npy')
train_labels = np.load('train_labels.npy')
test_pairs = np.load('test_pairs.npy')
test_labels = np.load('test_labels.npy')

# Dataset cho contrastive learning
class ContrastiveDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]
        return torch.tensor(pair[0], dtype=torch.float32), torch.tensor(pair[1], dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

train_dataset = ContrastiveDataset(train_pairs, train_labels)
test_dataset = ContrastiveDataset(test_pairs, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Mô hình contrastive learning
class ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim):
        super(ContrastiveModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)

# Loss function: Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = (1 - label) * torch.pow(euclidean_distance, 2) + \
               label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()

# Khởi tạo model, optimizer, và loss
embedding_dim = train_pairs.shape[2]
model = ContrastiveModel(embedding_dim)
criterion = ContrastiveLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for img_emb, text_emb, label in train_loader:
        optimizer.zero_grad()
        output1 = model(img_emb)
        output2 = model(text_emb)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# Đánh giá trên tập test
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for img_emb, text_emb, label in test_loader:
        output1 = model(img_emb)
        output2 = model(text_emb)
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        predictions = (euclidean_distance < 0.5).float()
        y_true.extend(label.numpy())
        y_pred.extend(predictions.numpy())

# Precision, Recall, F1-Score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
