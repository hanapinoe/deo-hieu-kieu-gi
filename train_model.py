import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Dataset cho Contrastive Learning
class ContrastiveDataset(Dataset):
    def __init__(self, pairs, labels, augment=False):
        self.pairs = pairs
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def augment_embedding(self, embedding):
        noise = np.random.normal(0, 0.01, embedding.shape)
        return embedding + noise

    def __getitem__(self, idx):
        img_emb, text_emb = self.pairs[idx]
        label = self.labels[idx]
        
        if self.augment:
            img_emb = self.augment_embedding(img_emb)
            text_emb = self.augment_embedding(text_emb)
        
        return (
            torch.tensor(img_emb, dtype=torch.float32),
            torch.tensor(text_emb, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


# Mô hình Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, img_embedding_dim, text_embedding_dim, output_dim=128):
        super(SiameseNetwork, self).__init__()
        self.img_transform = nn.Sequential(
            nn.Linear(img_embedding_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.text_transform = nn.Sequential(
            nn.Linear(text_embedding_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.shared_net = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

    def forward_once(self, x):
        return self.shared_net(x)

    def forward(self, img_input, text_input):
        img_embedding = self.img_transform(img_input)
        text_embedding = self.text_transform(text_input)
        output1 = self.forward_once(img_embedding)
        output2 = self.forward_once(text_embedding)
        return output1, output2


# Loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = (1 - label) * torch.pow(euclidean_distance, 2) + \
               label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()


# Load dữ liệu
train_pairs = np.load('train_pairs.npy', allow_pickle=True)
train_labels = np.load('train_labels.npy')
test_pairs = np.load('test_pairs.npy', allow_pickle=True)
test_labels = np.load('test_labels.npy')

train_dataset = ContrastiveDataset(train_pairs, train_labels)
test_dataset = ContrastiveDataset(test_pairs, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Khởi tạo mô hình
img_embedding_dim = 2048
text_embedding_dim = train_pairs[0][1].shape[0]
model = SiameseNetwork(img_embedding_dim, text_embedding_dim, output_dim=128)

criterion = ContrastiveLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện
num_epochs = 100
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

# Lưu mô hình
torch.save(model.state_dict(), 'siamese_model.pth')


print("Model has been saved.")
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
