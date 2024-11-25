import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Định nghĩa Dataset
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        scaler = StandardScaler()
        self.embeddings = scaler.fit_transform(embeddings)  # Chuẩn hóa dữ liệu

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32)

# Tải dữ liệu
embeddings = np.load('combined_embeddings.npy')
dataset = EmbeddingDataset(embeddings)

# Chia train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Định nghĩa Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Huấn luyện Autoencoder
input_dim = embeddings.shape[1]
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        encoded, decoded = model(batch)
        loss = criterion(decoded, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader)}")

print("Training complete.")

# Đánh giá mô hình
model.eval()
encoded_embeddings = []
with torch.no_grad():
    for batch in test_loader:
        encoded, _ = model(batch)
        encoded_embeddings.extend(encoded.numpy())

# Tính độ tương đồng cosine
encoded_embeddings = np.array(encoded_embeddings)
similarity_matrix = cosine_similarity(encoded_embeddings)

# Hiển thị trực quan bằng t-SNE
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(encoded_embeddings)

plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
plt.title("t-SNE Visualization of Encoded Embeddings")
plt.show()

# Lưu embeddings đã mã hóa
np.save('encoded_embeddings.npy', encoded_embeddings)
