import pandas as pd
import pymongo
import torch
import torchvision.transforms as transforms
from PIL import Image
import joblib
import base64
import os

# Kết nối MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["book_database"]
collection = db["books"]

# Tải mô hình ResNet và vectorizer
resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
resnet_model.eval()
vectorizer = joblib.load('vectorizer.pkl')

# Tiền xử lý ảnh cho mô hình ResNet-50
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Tạo embedding cho ảnh
def create_image_embedding(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        img_embedding = resnet_model(image_tensor).squeeze(0).numpy()
    return img_embedding.tolist()

# Tạo embedding cho văn bản
def create_text_embedding(title):
    return vectorizer.transform([title]).toarray()[0].tolist()

# Đọc dữ liệu từ CSV và lưu vào MongoDB
csv_file = "books_data.csv"
books_df = pd.read_csv(csv_file)

# Đường dẫn thư mục chứa ảnh
image_dir = os.path.join(os.getcwd(), 'static', 'images')

# Duyệt qua từng dòng trong DataFrame
for _, row in books_df.iterrows():
    title = row['Book Title']
    price = row['Price']
    image_filename = row['Image Path']  # Tên file ảnh từ CSV

    # Loại bỏ "/images" khỏi tên file ảnh
    image_filename = image_filename.replace('/images', '')
    # Đảm bảo rằng đường dẫn đúng đến thư mục static/images
    image_path = os.path.join(image_dir, image_filename)

    # Kiểm tra nếu tệp hình ảnh tồn tại
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy hình ảnh {image_path}")
        continue

    # Tạo embedding cho văn bản và hình ảnh
    text_embedding = create_text_embedding(title)
    image_embedding = create_image_embedding(image_path)

    # Đọc và chuyển ảnh thành Base64
    with open(image_path, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    # Tạo document để lưu vào MongoDB
    document = {
        "title": title,
        "price": price,
        "image_embedding": image_embedding,
        "text_embedding": text_embedding,
        "image_base64": image_base64,
        "image_url": image_filename  # Lưu tên file hình ảnh, không cần đường dẫn tuyệt đối
    }

    # Lưu document vào MongoDB
    collection.insert_one(document)

print("Dữ liệu đã được lưu vào MongoDB.")
