from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
from pymongo import MongoClient
import pytesseract
from PIL import Image
import torch.nn as nn
import tempfile
app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['book_search']
book_collection = db['books']

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


model = SiameseNetwork()
model.load_state_dict(torch.load('siamese_model.pth'))
model.eval()

# Load TfidfVectorizer
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



# Tải ResNet-50
resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()
# Trích xuất embedding ảnh
# Tiền xử lý ảnh với ResNet-50 và giảm kích thước xuống 128
# Trích xuất embedding ảnh
def create_image_embedding(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        features = resnet_model(image_tensor)  # Kích thước sẽ là (1, 2048)
    
    # Giảm kích thước từ 2048 xuống 128
    transform_layer = torch.nn.Linear(2048, 128)
    img_embedding = transform_layer(features.view(features.size(0), -1))  # (1, 2048) -> (1, 128)

    # Chuyển từ tensor sang NumPy array
    img_embedding = img_embedding.detach().numpy()

    # Kiểm tra NaN và xử lý
    if np.isnan(img_embedding).any():
        img_embedding = np.nan_to_num(img_embedding)

    return img_embedding.reshape(1, -1)


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

@app.route('/')
def index():
    return render_template('frontend.html')

# Tạo lớp chuyển đổi kích thước từ 128 lên 2048
img_expand_layer = torch.nn.Linear(128, 2048)  # Chuyển từ 128 lên 2048

@app.route('/search', methods=['POST'])
def search_books():
    try:
        # Kiểm tra xem người dùng nhập hình ảnh hay tiêu đề
        image = request.files.get('image')
        title = request.form.get('title')

        if not image and not title:
            return jsonify({"error": "Vui lòng cung cấp hình ảnh hoặc tiêu đề."}), 400

        # Khởi tạo các embedding mặc định
        img_embedding = None
        text_embedding = None

        if image:
            # Xử lý hình ảnh
            with tempfile.NamedTemporaryFile(delete=False) as temp_image:
                image.save(temp_image.name)

                # OCR để lấy văn bản từ hình ảnh
                ocr_text = pytesseract.image_to_string(temp_image.name, lang='eng').strip()

                # Trích xuất embedding ảnh và văn bản
                img_embedding = create_image_embedding(temp_image.name)
                text_embedding = create_text_embedding(ocr_text)

        if title:
            # Trích xuất embedding từ tiêu đề
            text_embedding = create_text_embedding(title)

        # Nếu không có text_embedding hoặc img_embedding, trả về lỗi
        if text_embedding is None:
            return jsonify({"error": "Không có embedding văn bản."}), 400

        # Tìm kiếm trong cơ sở dữ liệu
        books = list(book_collection.find())
        if not books:
            return jsonify({"error": "Không tìm thấy sách trong cơ sở dữ liệu."}), 404

        results = []

        for book in books:
            # Load embeddings từ cơ sở dữ liệu
            stored_img_embedding = torch.tensor(book['image_embedding'], dtype=torch.float32)
            stored_text_embedding = torch.tensor(book['text_embedding'], dtype=torch.float32)

            # Mở rộng kích thước của stored_img_embedding từ 128 lên 2048
            stored_img_embedding = img_expand_layer(stored_img_embedding.unsqueeze(0))  # (1, 128) -> (1, 2048)

            # Tiếp tục xử lý embedding ảnh và văn bản như bình thường
            with torch.no_grad():
                # Chuyển qua transform của model Siamese
                stored_img_embedding = model.img_transform(stored_img_embedding).squeeze(0)
                stored_text_embedding = model.text_transform(stored_text_embedding.unsqueeze(0)).squeeze(0)

                # Kiểm tra xem img_embedding có giá trị không, nếu không chỉ so sánh văn bản
                if img_embedding is None:
                    if text_embedding is not None:
                        text_similarity = torch.cosine_similarity(torch.tensor(text_embedding), stored_text_embedding.unsqueeze(0)).item()
                        total_similarity = text_similarity
                    else:
                        total_similarity = 0  # Nếu không có text_embedding, set similarity là 0
                else:
                    # Tính toán độ tương đồng giữa ảnh và văn bản
                    if text_embedding is not None:
                        img_similarity = torch.cosine_similarity(torch.tensor(img_embedding), stored_img_embedding.unsqueeze(0)).item()
                        text_similarity = torch.cosine_similarity(torch.tensor(text_embedding), stored_text_embedding.unsqueeze(0)).item()
                        total_similarity = (img_similarity + text_similarity) / 2
                    else:
                        total_similarity = 0  # Nếu không có text_embedding, set similarity là 0

            results.append({
                "title": book["title"],
                "price": book["price"],
                "relevance_score": float(total_similarity),
                "encoded_image": book["encoded_image"]
            })

        # Sắp xếp và trả về kết quả tương đồng cao nhất
        similarities = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
        return jsonify(similarities[:5])

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
