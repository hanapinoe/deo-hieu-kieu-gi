import os
import pandas as pd
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

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
            print(f"Cannot open image: {full_image_path}")

# Hàm tạo embedding văn bản
def create_text_embeddings(data):
    df = pd.DataFrame(data)
    df['price'] = df['price'].str.replace('₫', '').str.replace('.', '').astype(float)
    scaler = StandardScaler()
    df['price_scaled'] = scaler.fit_transform(df[['price']])

    vectorizer = TfidfVectorizer(max_features=100)
    title_embeddings = vectorizer.fit_transform(df['title']).toarray()

    return np.hstack((title_embeddings, df[['price_scaled']].values))

# Hàm tạo embedding ảnh
def create_image_embeddings(data):
    model = models.resnet50(pretrained=True)
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

# Kết hợp embedding văn bản và embedding ảnh
combined_embeddings = np.hstack((text_embeddings, image_embeddings))

# Lưu kết quả thành file .npy
np.save('combined_embeddings.npy', combined_embeddings)

print("Embedding kết hợp đã được lưu thành công.")
