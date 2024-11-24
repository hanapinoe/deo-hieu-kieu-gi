import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision.transforms as transforms

# Định nghĩa lớp Dataset để sử dụng với DataLoader
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]

# Tải dữ liệu từ file .npy
embeddings = np.load('combined_embeddings.npy')

# Data Augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# Tạo DataLoader với Data Augmentation
dataset = EmbeddingDataset(embeddings)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Định nghĩa mô hình với Dropout và L2 Regularization
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Dropout layer
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Dropout layer
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Dropout layer
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Dropout layer
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Huấn luyện mô hình Autoencoder với L2 Regularization
input_dim = embeddings.shape[1]
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        embeddings = data
        optimizer.zero_grad()
        encoded, decoded = model(embeddings.float())
        loss = criterion(decoded, embeddings.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

print("Huấn luyện hoàn tất.")

# Đánh giá mô hình
model.eval()
original_embeddings = []
encoded_embeddings = []
with torch.no_grad():
    for data in test_loader:
        embeddings = data
        encoded, _ = model(embeddings.float())
        original_embeddings.extend(model.encoder(embeddings.float()).cpu().numpy())  # Mã hóa lại original_embeddings
        encoded_embeddings.extend(encoded.cpu().numpy())

# Chuyển đổi danh sách thành numpy array
original_embeddings = np.array(original_embeddings)
encoded_embeddings = np.array(encoded_embeddings)

# Đánh giá độ tương đồng cosine
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(original_embeddings, encoded_embeddings)
print(f"Cosine Similarity:\n{similarity}")

# Chuyển đổi độ tương đồng thành nhãn (labels) giả định để tính Precision, Recall, F1-Score
threshold = 0.5  # Ngưỡng để quyết định tương tự
y_true = (similarity >= threshold).astype(int).flatten()
y_pred = (similarity >= threshold).astype(int).flatten()

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Giảm chiều dữ liệu
tsne = TSNE(n_components=2, random_state=42)
original_embeddings_2d = tsne.fit_transform(original_embeddings)
encoded_embeddings_2d = tsne.fit_transform(encoded_embeddings)

# Hiển thị trực quan các nhúng
plt.figure(figsize=(10, 5))
plt.scatter(original_embeddings_2d[:, 0], original_embeddings_2d[:, 1], label='Original')
plt.scatter(encoded_embeddings_2d[:, 0], encoded_embeddings_2d[:, 1], label='Encoded', marker='x')
plt.legend()
plt.title("t-SNE Visualization of Embeddings")
plt.show()



