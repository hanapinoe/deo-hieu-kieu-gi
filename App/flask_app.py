from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import os
import joblib
import pytesseract  # Thư viện OCR
from flask_cors import CORS

# Đặt đường dẫn đến tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)  # Cho phép CORS

# Kết nối với MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['bookstore']
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

# Tải mô hình và vectorizer
img_embedding_dim = 2048
text_embedding_dim = 101
model = SiameseNetwork(img_embedding_dim, text_embedding_dim, output_dim=128)
model.load_state_dict(torch.load('siamese_model.pth'))
model.eval()

vectorizer = joblib.load('vectorizer.pkl')

# Tiền xử lý ảnh
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Trích xuất văn bản từ ảnh bằng OCR
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text.strip()

# Tạo embedding cho văn bản
def create_text_embedding(title):
    return vectorizer.transform([title]).toarray()[0]

# Tạo embedding cho ảnh
def create_image_embedding(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        features = model(image_tensor)
    
    img_embedding = features.view(features.size(0), -1)  # Đảm bảo kích thước đúng
    return img_embedding.numpy()

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/search', methods=['POST'])
def search_books():
    if not os.path.exists('temp'):
        os.makedirs('temp')

    image = request.files.get('image')
    title = request.form.get('title')

    if not image and not title:
        return jsonify({"error": "Vui lòng cung cấp ảnh hoặc tiêu đề."}), 400

    input_embedding = None

    if image:
        image_path = f"temp/{image.filename}"
        image.save(image_path)

        # Sử dụng OCR trước, nếu thất bại dùng embedding ảnh
        extracted_text = extract_text_from_image(image_path)
        if extracted_text:
            input_embedding = create_text_embedding(extracted_text)
        else:
            input_embedding = create_image_embedding(image_path)
    elif title:
        input_embedding = create_text_embedding(title)

    books = list(book_collection.find())
    results = []

    for book in books:
        if input_embedding is not None:
            if image:  # So sánh bằng ảnh
                embedding = np.array(book.get('image_embedding', []))
            elif title:  # So sánh bằng tiêu đề
                embedding = np.array(book.get('text_embedding', []))

            similarity = cosine_similarity([input_embedding], [embedding])[0][0]
            results.append({
                "title": book["title"],
                "price": book["price"],
                "image_url": book["image_url"],
                "similarity": similarity
            })

    # Sắp xếp theo độ tương đồng
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:5]

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
