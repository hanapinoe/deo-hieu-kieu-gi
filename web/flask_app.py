from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import tempfile
import joblib
import pytesseract
from flask_cors import CORS

# Đặt đường dẫn đến Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Kết nối với MongoDB
try:
    client = MongoClient('mongodb://localhost:27017/')
    client.admin.command('ping')
    db = client['book_search']
    book_collection = db['books']
except Exception as e:
    print(f"Không thể kết nối với MongoDB: {str(e)}")
    exit(1)

# Tải vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Tải ResNet-50
resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()


def preprocess_title(title):
    """
    Chuẩn hóa tiêu đề người dùng nhập.
    """
    import re
    # Chuyển về chữ thường
    title = title.lower()
    # Loại bỏ khoảng trắng thừa
    title = title.strip()
    # Loại bỏ dấu câu và ký tự đặc biệt (nếu cần)
    title = re.sub(r'[^\w\s]', '', title)
    return title

def create_text_embedding(title):
    """
    Tạo embedding từ tiêu đề văn bản sau khi chuẩn hóa.
    """
    # Tiền xử lý tiêu đề
    title = preprocess_title(title)

    # Tạo embedding
    text_embedding = vectorizer.transform([title]).toarray()
    text_embedding = torch.tensor(text_embedding, dtype=torch.float)
    transform_layer = torch.nn.Linear(text_embedding.size(1), 128)

    with torch.no_grad():
        text_embedding = transform_layer(text_embedding).detach().numpy()

    # Kiểm tra NaN và xử lý
    if np.isnan(text_embedding).any():
        text_embedding = np.nan_to_num(text_embedding)

    return text_embedding.reshape(1, -1)

# Tiền xử lý ảnh
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Trích xuất embedding ảnh
def create_image_embedding(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        features = resnet_model(image_tensor)
    img_embedding = features.view(features.size(0), -1).cpu().numpy()

    # Kiểm tra NaN và xử lý
    if np.isnan(img_embedding).any():
        img_embedding = np.nan_to_num(img_embedding)

    return img_embedding.reshape(1, -1)

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/search', methods=['POST'])
def search_books():
    try:
        # Kiểm tra xem người dùng nhập hình ảnh hay tiêu đề
        image = request.files.get('image')
        title = request.form.get('title')

        if not image and not title:
            return jsonify({"error": "Vui lòng cung cấp hình ảnh hoặc tiêu đề."}), 400

        if image:
            # Xử lý hình ảnh
            with tempfile.NamedTemporaryFile(delete=False) as temp_image:
                image.save(temp_image.name)

                # OCR để lấy văn bản từ hình ảnh
                ocr_text = pytesseract.image_to_string(temp_image.name, lang='eng').strip()

                # Trích xuất embedding ảnh và văn bản
                img_embedding = create_image_embedding(temp_image.name)
                text_embedding = create_text_embedding(ocr_text)

        elif title:
            # Trích xuất embedding từ tiêu đề
            text_embedding = create_text_embedding(title)

        # Tìm kiếm trong cơ sở dữ liệu
        books = list(book_collection.find())
        if not books:
            return jsonify({"error": "Không tìm thấy sách trong cơ sở dữ liệu."}), 404

        results = []
        for book in books:
            stored_img_embedding = np.array(book['image_embedding']).reshape(1, -1)
            stored_text_embedding = np.array(book['text_embedding']).reshape(1, -1)

            # Kiểm tra NaN và xử lý
            if np.isnan(stored_img_embedding).any():
                stored_img_embedding = np.nan_to_num(stored_img_embedding)
            if np.isnan(stored_text_embedding).any():
                stored_text_embedding = np.nan_to_num(stored_text_embedding)

            # Tính độ tương đồng cosine
            if image:
                img_similarity = cosine_similarity(img_embedding, stored_img_embedding)[0][0]
            else:
                img_similarity = 0  # Không tính điểm ảnh nếu không có hình ảnh

            text_similarity = cosine_similarity(text_embedding, stored_text_embedding)[0][0]

            # Tổng hợp điểm tương đồng
            overall_similarity = (img_similarity + text_similarity) / (2 if image else 1)

            results.append({
                "title": book["title"],
                "price": book["price"],
                "relevance_score": float(overall_similarity),
                "encoded_image": book["encoded_image"]
            })

        # Sắp xếp kết quả theo relevance_score và lấy 5 kết quả đầu tiên
        results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:5]

        return jsonify(results),print(f"Số lượng tài liệu truy xuất: {len(books)}")

    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return jsonify({"error": f"Lỗi: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)