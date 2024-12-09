import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Dataset class
class ContrastiveDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_emb, text_emb = self.pairs[idx]
        label = self.labels[idx]
        return (
            torch.tensor(img_emb, dtype=torch.float32),
            torch.tensor(text_emb, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),  # Ensure label is long
        )

# Siamese Network class
class SiameseNetwork(nn.Module):
    def __init__(self, img_embedding_dim=2048, text_embedding_dim=128, output_dim=128):
        super(SiameseNetwork, self).__init__()
        self.img_transform = nn.Sequential(
            nn.Linear(img_embedding_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        self.text_transform = nn.Sequential(
            nn.Linear(text_embedding_dim, output_dim),  # Đầu vào là 128 (TF-IDF vector)
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        self.shared_net = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
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


# Contrastive Loss class
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = (1 - label) * torch.pow(euclidean_distance, 2) + \
               label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()

# Load data
train_pairs = np.load('train_pairs.npy', allow_pickle=True)
train_labels = np.load('train_labels.npy')
test_pairs = np.load('test_pairs.npy', allow_pickle=True)
test_labels = np.load('test_labels.npy')

train_dataset = ContrastiveDataset(train_pairs, train_labels)
test_dataset = ContrastiveDataset(test_pairs, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_embedding_dim = 2048
text_embedding_dim = train_pairs[0][1].shape[0]
model = SiameseNetwork(img_embedding_dim, text_embedding_dim, output_dim=128).to(device)

criterion = ContrastiveLoss(margin=1.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for img_emb, text_emb, label in train_loader:
        img_emb, text_emb, label = img_emb.to(device), text_emb.to(device), label.to(device)
        
        optimizer.zero_grad()
        output1, output2 = model(img_emb, text_emb)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'siamese_model.pth')
print("Model has been saved.")

# Evaluate model
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for img_emb, text_emb, label in test_loader:
        img_emb, text_emb, label = img_emb.to(device), text_emb.to(device), label.to(device)
        
        output1, output2 = model(img_emb, text_emb)
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        predictions = (euclidean_distance < 0.7).float()  # Adjust threshold
        y_true.extend(label.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
