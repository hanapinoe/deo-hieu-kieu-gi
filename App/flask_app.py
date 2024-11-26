from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import os
import joblib

# Khởi tạo Flask app
app = Flask(__name__)

# Kết nối với MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Cập nhật với chuỗi kết nối MongoDB của bạn
db = client['book_database']
book_collection = db['books']

# Tải mô hình Siamese đã huấn luyện
class SiameseNetwork(torch.nn.Module):
    def __init__(self, img_embedding_dim, text_embedding_dim, output_dim=128):
        super(SiameseNetwork, self).__init__()
        self.img_transform = torch.nn.Linear(img_embedding_dim, output_dim)
        self.text_transform = torch.nn.Linear(text_embedding_dim, output_dim)
        self.shared_net = torch.nn.Sequential(
            torch.nn.Linear(output_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU()
        )

    def forward_once(self, x):
        return self.shared_net(x)

    def forward(self, img_input, text_input):
        img_embedding = self.img_transform(img_input)
        text_embedding = self.text_transform(text_input)
        output1 = self.forward_once(img_embedding)
        output2 = self.forward_once(text_embedding)
        return output1, output2

# Tải mô hình đã huấn luyện
img_embedding_dim = 2048  # Kích thước embedding ảnh từ ResNet-50
text_embedding_dim = 100  # Kích thước embedding văn bản từ vectorizer
model = SiameseNetwork(img_embedding_dim, text_embedding_dim, output_dim=128)
model.load_state_dict(torch.load('siamese_model.pth'))  # Tải mô hình Siamese đã huấn luyện
model.eval()

# Tiền xử lý ảnh cho mô hình ResNet-50
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Tạo embedding cho văn bản (tiêu đề sách)
def create_text_embedding(title):
    # Tải vectorizer đã huấn luyện từ file 'vectorizer.pkl'
    vectorizer = joblib.load('vectorizer.pkl')  # Tải vectorizer đã lưu
    text_embedding = vectorizer.transform([title]).toarray()[0]
    return text_embedding

# Tạo embedding cho ảnh
def create_image_embedding(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        img_embedding, _ = model(image_tensor.view(-1, img_embedding_dim))
    return img_embedding.numpy()

# Tìm kiếm sách dựa trên ảnh hoặc tiêu đề
@app.route('/search', methods=['POST'])
def search_books():
    # Đảm bảo thư mục temp tồn tại
    if not os.path.exists('temp'):
        os.makedirs('temp')

    # Lấy dữ liệu từ yêu cầu
    image = request.files.get('image')
    title = request.form.get('title')

    if not image and not title:
        return jsonify({"error": "Vui lòng cung cấp ảnh hoặc tiêu đề."}), 400

    # Xử lý dữ liệu đầu vào
    input_embedding = None
    if image:
        image_path = f"temp/{image.filename}"
        image.save(image_path)
        input_embedding = create_image_embedding(image_path)
    elif title:
        input_embedding = create_text_embedding(title)

    # Lấy danh sách sách từ MongoDB
    books = list(book_collection.find())
    book_list = []
    for book in books:
        if 'embedding' in book and 'title' in book:
            book_list.append({
                "title": book['title'],
                "embedding": np.array(book['embedding']),
                "metadata": book.get('metadata', 'N/A')
            })

    # Tính toán độ tương đồng cosine
    embeddings = np.array([book['embedding'] for book in book_list])
    similarities = cosine_similarity([input_embedding], embeddings)[0]
    top_matches = sorted(zip(book_list, similarities), key=lambda x: x[1], reverse=True)[:5]

    # Trả về 5 kết quả có độ tương đồng cao nhất
    response = [{"title": match[0]['title'], "similarity": match[1], "metadata": match[0]['metadata']} for match in top_matches]
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
