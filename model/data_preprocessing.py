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
    title = row['title']       # Cột chứa tiêu đề sách
    price = row['price']            # Cột chứa giá sách

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

# Cập nhật hàm tạo embedding văn bản
def create_text_embeddings(data):
    df = pd.DataFrame(data)
    df['price'] = (
        df['price']
        .astype(str)  # Đảm bảo tất cả giá trị là chuỗi
        .str.replace('₫', '', regex=True)  # Loại bỏ ký tự ₫
        .str.replace('đ', '', regex=True)  # Loại bỏ ký tự đ
        .str.replace('.', '', regex=True)  # Loại bỏ dấu chấm phân cách
        .str.strip()  # Loại bỏ khoảng trắng thừa (nếu có)
    )

    # Chuyển sang kiểu float
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Chuẩn hóa giá sách
    scaler = StandardScaler()
    df['price_scaled'] = scaler.fit_transform(df[['price']])

    # Tạo TfidfVectorizer hoặc tải nếu đã lưu trước đó
    vectorizer_path = 'vectorizer.pkl'
    transform_layer_path = 'transform_layer.pth'
    
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        vectorizer = TfidfVectorizer(max_features=128)
        vectorizer.fit(df['title'])
        joblib.dump(vectorizer, vectorizer_path)
    
    title_embeddings = vectorizer.transform(df['title']).toarray()

    # Kết hợp title_embeddings với giá sách (giữ kích thước nhất quán với pipeline)
    text_embeddings = np.hstack((title_embeddings, df[['price_scaled']].values))
    
    if os.path.exists(transform_layer_path):
        transform_layer = torch.nn.Linear(text_embeddings.shape[1], 128)
        transform_layer.load_state_dict(torch.load(transform_layer_path))
    else:
        transform_layer = torch.nn.Linear(text_embeddings.shape[1], 128)
        torch.save(transform_layer.state_dict(), transform_layer_path)
    
    # Chuyển đổi kích thước từ 129 -> 128 bằng lớp Linear
    text_embeddings = torch.tensor(text_embeddings, dtype=torch.float)
    text_embeddings = transform_layer(text_embeddings).detach().numpy()

    return text_embeddings

# Hàm tạo embedding ảnh
def create_image_embeddings(data):
    from torchvision.models import ResNet50_Weights
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Loại bỏ lớp cuối (fully connected layer)
    model = torch.nn.Sequential(*list(model.children())[:-1])

    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_image_embedding(image):
        image = preprocess(image).unsqueeze(0)  # Thêm batch dimension
        with torch.no_grad():
            features = model(image)  # Đầu ra của ResNet50 sau lớp pooling
        return features.view(-1).numpy()  # Flatten vector từ (1, 2048) thành (2048,)

    return np.array([get_image_embedding(d['image']) for d in data])


# Hàm mã hóa ảnh thành base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image

# Tạo embedding văn bản và ảnh
text_embeddings = create_text_embeddings(dataset)
image_embeddings = create_image_embeddings(dataset)

# Đảm bảo kích thước đồng nhất
assert text_embeddings.shape[1] == 128, "Text embeddings phải có kích thước 128"
assert image_embeddings.shape[1] == 2048, "Image embeddings phải có kích thước 2048"

# Lưu kết quả vào MongoDB
for idx, row in enumerate(dataset):
    # Mã hóa ảnh thành base64
    encoded_image = encode_image_to_base64(row['image_path'])

    # Tạo document
    document = {
        "title": row['title'],
        "price": row['price'],
        "text_embedding": text_embeddings[idx].tolist(),  # Chuyển sang danh sách để lưu vào MongoDB
        "image_embedding": image_embeddings[idx].tolist(),  # Chuyển sang danh sách
        "encoded_image": encoded_image  # Lưu ảnh mã hóa base64
    }
    
    # Chèn document vào MongoDB
    collection.insert_one(document)

print("Embedding đã được lưu vào MongoDB thành công.")