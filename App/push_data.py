import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50, ResNet50_Weights
import random
from pymongo import MongoClient

# Đọc dữ liệu từ file CSV
data = pd.read_csv('books_data.csv')
dataset = []

# Duyệt qua từng hàng trong DataFrame
for _, row in data.iterrows():
    image_path = row['Image Path']  # Cột chứa đường dẫn ảnh
    title = row['Book Title']       # Cột chứa tiêu đề sách
    price = row['Price']            # Cột chứa giá sách

    # Xử lý đường dẫn ảnh
    if image_path.startswith('images/'):
        image_path = image_path[len('images/'):]
    full_image_path = os.path.join('./static/images', image_path)
    
    # Kiểm tra sự tồn tại của file ảnh
    if os.path.exists(full_image_path):
        try:
            # Mở ảnh và chuyển đổi thành RGB
            image = Image.open(full_image_path).convert('RGB')
            dataset.append({'image': image, 'title': title, 'price': price})
        except IOError as e:
            print(f"Error opening image {full_image_path}: {e}")

# Hàm tạo embedding văn bản
def create_text_embeddings(data):
    df = pd.DataFrame(data)

    # Kiểm tra và xử lý giá trị thiếu trong cột `price`
    if 'price' in df.columns:
        try:
            df['price'] = df['price'].str.replace(r'[₫.,]', '', regex=True).astype(float)
        except Exception as e:
            print(f"Error processing price column: {e}")
            df['price'] = 0.0  # Giá trị mặc định nếu không xử lý được

        # Thay thế các giá trị thiếu bằng giá trị trung bình của cột `price`
        df['price'].fillna(df['price'].mean(), inplace=True)
    else:
        df['price'] = 0.0  # Thêm cột `price` với giá trị mặc định nếu không tồn tại

    # Chuẩn hóa giá
    scaler = StandardScaler()
    df['price_scaled'] = scaler.fit_transform(df[['price']])

    # Tạo embedding tiêu đề bằng TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)
    title_embeddings = vectorizer.fit_transform(df['title']).toarray()

    return np.hstack((title_embeddings, df[['price_scaled']].values))

# Hàm tạo embedding ảnh
def create_image_embeddings(data):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Lấy layer trước classifier
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_image_embedding(idx, image):
        try:
            image_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                features = model(image_tensor)
            return features.squeeze().numpy()
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")
            return np.zeros(2048)

    return [get_image_embedding(idx, d['image']) for idx, d in enumerate(data)]

# Tạo embedding văn bản và ảnh
text_embeddings = create_text_embeddings(dataset)
image_embeddings = create_image_embeddings(dataset)

# Chuẩn hóa embedding một lần
scaler = StandardScaler()
text_embeddings = scaler.fit_transform(text_embeddings)
image_embeddings = scaler.fit_transform(image_embeddings)

# Ghép thông tin embedding
valid_dataset = []
for idx, d in enumerate(dataset):
    if idx < len(text_embeddings) and idx < len(image_embeddings):
        valid_dataset.append({
            'title': d['title'],
            'price': d['price'],
            'text_embedding': text_embeddings[idx].tolist(),  # Convert to list for MongoDB
            'image_embedding': image_embeddings[idx].tolist()  # Convert to list for MongoDB
        })

# Hàm đẩy dữ liệu lên MongoDB
def push_data_to_mongodb(data, db_name='book_db', collection_name='book_embeddings'):
    client = MongoClient('mongodb://localhost:27017/')  # Thay thế bằng URL của MongoDB nếu cần
    db = client[db_name]
    collection = db[collection_name]
    
    # Chèn dữ liệu
    collection.insert_many(data)
    print(f"Successfully inserted {len(data)} records into the {collection_name} collection in the {db_name} database.")

# Đẩy dữ liệu lên MongoDB
push_data_to_mongodb(valid_dataset)

print("Data for contrastive learning has been prepared and pushed to MongoDB successfully.")
