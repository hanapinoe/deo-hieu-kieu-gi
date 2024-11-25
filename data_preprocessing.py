import os
import pandas as pd
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50, ResNet50_Weights

# Đọc dữ liệu từ file CSV
data = pd.read_csv('books_data.csv')
dataset = []

# Duyệt qua từng hàng trong DataFrame
for _, row in data.iterrows():
    image_path = row['Image Path']  # Cột chứa đường dẫn ảnh
    title = row['Book Title']       # Cột chứa tiêu đề sách
    price = row['Price']            # Cột chứa giá sách

    # Xóa phần 'images/' nếu có trong image_path và thêm vào đường dẫn tĩnh
    if image_path.startswith('images/'):
        image_path = image_path[len('images/'):]
    full_image_path = os.path.join('./static/images', image_path)
    
    # Kiểm tra sự tồn tại của file ảnh
    if os.path.exists(full_image_path):
        try:
            # Mở ảnh bằng Pillow
            image = Image.open(full_image_path).convert('RGB')  # Chuyển đổi thành ảnh RGB
            # Thêm thông tin vào dataset
            dataset.append({'image': image, 'title': title, 'price': price})
        except IOError:
            continue  # Bỏ qua ảnh lỗi

# Hàm tạo embedding văn bản
def create_text_embeddings(data):
    df = pd.DataFrame(data)
    df['price'] = df['price'].str.replace(r'[₫.]', '', regex=True).astype(float)
    scaler = StandardScaler()
    df['price_scaled'] = scaler.fit_transform(df[['price']])

    vectorizer = TfidfVectorizer(max_features=100)
    title_embeddings = vectorizer.fit_transform(df['title']).toarray()

    return np.hstack((title_embeddings, df[['price_scaled']].values))

# Hàm tạo embedding ảnh

def create_image_embeddings(data):
    # Sử dụng trọng số đúng
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Hoặc ResNet50_Weights.DEFAULT
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Lấy layer trước classifier
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_image_embedding(image):
        image = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = model(image)
        return features.squeeze().numpy()

    return np.array([get_image_embedding(d['image']) for d in data])


# Tạo embedding văn bản và ảnh
text_embeddings = create_text_embeddings(dataset)
image_embeddings = create_image_embeddings(dataset)

# Normalize từng embedding
text_embeddings = StandardScaler().fit_transform(text_embeddings)
image_embeddings = StandardScaler().fit_transform(image_embeddings)

# Ghép thông tin embedding
for idx, d in enumerate(dataset):
    d['text_embedding'] = text_embeddings[idx]
    d['image_embedding'] = image_embeddings[idx]

# Tạo các cặp dữ liệu (positive/negative pairs)
def create_pairs(data):
    pairs = []
    labels = []
    for i, item_a in enumerate(data):
        for j, item_b in enumerate(data):
            if i == j:
                # Positive Pair: Ảnh và text tương ứng
                pairs.append((item_a['image_embedding'], item_a['text_embedding']))
                labels.append(1)
            else:
                # Negative Pair: Ảnh và text không tương ứng
                pairs.append((item_a['image_embedding'], item_b['text_embedding']))
                labels.append(0)
    return np.array(pairs), np.array(labels)

pairs, labels = create_pairs(dataset)

# Chia dữ liệu train/test
train_pairs, test_pairs, train_labels, test_labels = train_test_split(pairs, labels, test_size=0.2, random_state=42)

# Lưu dữ liệu train/test để sử dụng
np.save('train_pairs.npy', train_pairs)
np.save('train_labels.npy', train_labels)
np.save('test_pairs.npy', test_pairs)
np.save('test_labels.npy', test_labels)

print("Data for contrastive learning has been prepared successfully.")
