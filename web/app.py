from flask import Flask, request, jsonify, render_template
from siamese_model import SiameseModel
from book_search import BookSearch
from mongo_client import CustomMongoClient
import pytesseract
from PIL import Image
import numpy as np

app = Flask(__name__)

# Đảm bảo bạn đã có đường dẫn đến mô hình và vectorizer
MODEL_PATH = 'D:/Project/model1/siamese_model.pth'  # Đảm bảo đường dẫn chính xác
VECTORIZER_PATH = 'D:/Project/model1/vectorizer.pkl'  # Đảm bảo đường dẫn chính xác

# Khởi tạo mô hình Siamese
siamese_model = SiameseModel(model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH)

# Khởi tạo đối tượng tìm kiếm sách
book_search = BookSearch(model=siamese_model)

# Kết nối MongoDB thông qua MongoClient (sửa kết nối)
mongo_client = CustomMongoClient('mongodb://localhost:27017/')
db = mongo_client.get_db()  # Lấy database 'book_search'
book_collection = db['books']  # Lấy collection 'books'

def extract_text_from_image(image_path):
    """Nhận diện văn bản từ hình ảnh sử dụng OCR (Tesseract)"""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/search', methods=['POST'])
def search_books():
    data = request.form
    image_path = data.get('image_path', None)
    title_query = data.get('title', None)

    if not image_path and not title_query:
        return jsonify({"error": "Cần cung cấp ít nhất một trong hai: 'image_path' hoặc 'title'"}), 400

    # Nếu có hình ảnh, sử dụng OCR để nhận diện văn bản từ hình ảnh
    if image_path:
        extracted_text = extract_text_from_image(image_path)
        title_query = title_query or extracted_text  # Nếu không có tiêu đề từ truy vấn, sử dụng OCR

    img_embedding = None
    text_embedding = None

    if image_path:
        img_embedding = siamese_model.create_image_embedding(image_path)
    
    if title_query:
        text_embedding = siamese_model.create_text_embedding(title_query)

    books = book_collection.find()  # Lấy tất cả sách từ collection 'books'
    books = list(books)  # Chuyển dữ liệu từ MongoDB thành danh sách

    # Tạo embedding cho các sách trong cơ sở dữ liệu
    for book in books:
        book['image_embedding'] = np.array(book['image_embedding']).reshape(1, -1)
        book['text_embedding'] = np.array(book['text_embedding']).reshape(1, -1)

    # Tìm kiếm sách dựa trên embedding
    results = book_search.search_books(books, img_embedding, text_embedding)

    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True)
