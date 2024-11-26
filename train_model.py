import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load dữ liệu
train_pairs = np.load('train_pairs.npy', allow_pickle=True)  # Đảm bảo dtype là object
train_labels = np.load('train_labels.npy')
test_pairs = np.load('test_pairs.npy', allow_pickle=True)
test_labels = np.load('test_labels.npy')

# Dataset cho contrastive learning
class ContrastiveDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_emb, text_emb = self.pairs[idx]  # Tách cặp embedding ảnh và văn bản
        label = self.labels[idx]
        return (
            torch.tensor(img_emb, dtype=torch.float32),
            torch.tensor(text_emb, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

# Tạo Dataset và DataLoader
train_dataset = ContrastiveDataset(train_pairs, train_labels)
test_dataset = ContrastiveDataset(test_pairs, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Mô hình Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward_once(self, x):
        return self.shared_net(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

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

# Khởi tạo mô hình, loss, và optimizer
embedding_dim = train_pairs[0][0].shape[0]  # Lấy số chiều từ embedding ảnh
model = SiameseNetwork(embedding_dim)
criterion = ContrastiveLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for img_emb, text_emb, label in train_loader:
        optimizer.zero_grad()
        output1, output2 = model(img_emb, text_emb)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Đánh giá trên tập test
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for img_emb, text_emb, label in test_loader:
        output1, output2 = model(img_emb, text_emb)
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        predictions = (euclidean_distance < 0.5).float()
        y_true.extend(label.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

# Tính toán Precision, Recall, F1-Score và Accuracy
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
