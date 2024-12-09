import os
import pandas as pd
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient
import base64
import joblib

# Kết nối MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['book_search']
collection = db['books']

# Đọc dữ liệu từ file CSV
data = pd.read_csv('meta_data.csv')
dataset = []

# Duyệt qua từng hàng trong DataFrame
for _, row in data.iterrows():
    image_path = row['image path']  # Cột chứa đường dẫn ảnh
    title = row['title']           # Cột chứa tiêu đề sách
    price = row['price']           # Cột chứa giá sách

    # Xóa phần 'images/' nếu có trong image_path và thêm vào đường dẫn tĩnh
    if image_path.startswith('books/'):
        image_path = image_path[len('books/'):]
    full_image_path = os.path.join('./static/images', image_path)
    
    # Kiểm tra sự tồn tại của file ảnh
    if os.path.exists(full_image_path):
        try:
            # Mở ảnh bằng Pillow
            image = Image.open(full_image_path).convert('RGB')  # Chuyển đổi thành ảnh RGB
            # Thêm thông tin vào dataset
            dataset.append({'image': image, 'title': title, 'price': price, 'image_path': full_image_path})
        except IOError:
            print(f"Cannot open image: {full_image_path}")

# Tạo và lưu transform layers
image_transform_layer_path = 'image_transform_layer.pth'
text_transform_layer_path = 'text_transform_layer.pth'

# Hàm tạo transform layer
def create_transform_layer(input_dim, output_dim, path):
    layer = torch.nn.Linear(input_dim, output_dim)
    if os.path.exists(path):
        layer.load_state_dict(torch.load(path))
    else:
        torch.save(layer.state_dict(), path)
    return layer

def create_text_embeddings(data, vectorizer_path='vectorizer.pkl'):
    # Nếu data là danh sách, chuyển đổi nó thành DataFrame
    if isinstance(data, list):
        data = pd.DataFrame(data)

    # Đảm bảo cột 'title' tồn tại
    if 'title' not in data.columns:
        raise ValueError("Dữ liệu không chứa cột 'title'.")

    # Tạo hoặc tải vectorizer
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        vectorizer = TfidfVectorizer(max_features=128)
        vectorizer.fit(data['title'])
        joblib.dump(vectorizer, vectorizer_path)

    # Tạo embedding từ TF-IDF vectorizer
    text_embeddings = vectorizer.transform(data['title']).toarray()
    return text_embeddings

# Hàm tạo embedding ảnh
def create_image_embeddings(data):
    # Tải ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Loại bỏ FC layer
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
        return features.view(-1).numpy()

    image_embeddings = np.array([get_image_embedding(d['image']) for d in data])

    # Áp dụng transform layer
    image_transform_layer = create_transform_layer(2048, 128, image_transform_layer_path)
    image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)
    image_embeddings = image_transform_layer(image_embeddings).detach().numpy()

    return image_embeddings

# Hàm mã hóa ảnh
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image

# Tạo embedding văn bản và ảnh
text_embeddings = create_text_embeddings(dataset)
image_embeddings = create_image_embeddings(dataset)

# Lưu vào MongoDB
for idx, row in enumerate(dataset):
    encoded_image = encode_image_to_base64(row['image_path'])
    document = {
        "title": row['title'],
        "price": row['price'],
        "text_embedding": text_embeddings[idx].tolist(),
        "image_embedding": image_embeddings[idx].tolist(),
        "encoded_image": encoded_image
    }
    collection.insert_one(document)

print("Embedding đã được lưu vào MongoDB thành công.")
