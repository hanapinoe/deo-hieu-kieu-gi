from flask import Flask, request, jsonify, render_template
from siamese_model import SiameseModel  # Mô hình Siamese
from book_search import BookSearch  # Lớp tìm kiếm sách
from mongo_client import CustomMongoClient  # Kết nối MongoDB
from PIL import Image
import os
import numpy as np

app = Flask(__name__)

# Đường dẫn mô hình và vectorizer
MODEL_PATH = 'D:/Project/model/siamese_model.pth'
VECTORIZER_PATH = 'D:/Project/model/vectorizer.pkl'

# Thư mục tạm để lưu file ảnh
TEMP_IMAGE_FOLDER = 'temp_images'
if not os.path.exists(TEMP_IMAGE_FOLDER):
    os.makedirs(TEMP_IMAGE_FOLDER)

# Giới hạn kích thước file upload (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Khởi tạo mô hình Siamese
try:
    siamese_model = SiameseModel(model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH)
    print("Mô hình Siamese đã được tải thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình Siamese: {e}")

# Khởi tạo đối tượng tìm kiếm sách
book_search = BookSearch(model=siamese_model)

# Kết nối MongoDB thông qua CustomMongoClient
try:
    mongo_client = CustomMongoClient('mongodb://localhost:27017/')
    db = mongo_client.get_db()  # Lấy database 'book_search'
    print("Kết nối MongoDB thành công!")
except Exception as e:
    print(f"Lỗi khi kết nối MongoDB: {e}")


@app.route('/')
def home():
    """Trang chủ - Giao diện frontend."""
    return render_template('frontend.html')

# Biến toàn cục để lưu trạng thái embedding
img_embedding = None
text_embedding = None

@app.route('/reset', methods=['POST'])
def reset_state():
    """Reset trạng thái của server, xóa embedding và dữ liệu tạm."""
    global img_embedding, text_embedding

    try:
        # Reset các embedding
        img_embedding = None
        text_embedding = None

        # Xóa toàn bộ file trong thư mục tạm (nếu có)
        for file_name in os.listdir(TEMP_IMAGE_FOLDER):
            file_path = os.path.join(TEMP_IMAGE_FOLDER, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # In ra log để xác nhận đã reset
        print("Đã reset trạng thái embedding và xóa dữ liệu tạm.")
        return jsonify({"message": "Trạng thái đã được reset thành công."}), 200
    except Exception as e:
        return jsonify({"error": f"Lỗi khi reset trạng thái: {str(e)}"}), 500


@app.route('/search', methods=['POST'])
def search_books():
    global img_embedding, text_embedding

    try:
        # Lấy dữ liệu từ request
        data = request.form
        file = request.files.get('image_path')  # Lấy file từ request
        title_query = data.get('title', None)  # Lấy tiêu đề từ request

        # Kiểm tra nếu cả hình ảnh và tiêu đề đều không được cung cấp
        if not file and not title_query:
            return jsonify({"error": "Cần cung cấp ít nhất một trong hai: 'image_path' hoặc 'title'"}), 400

        # Tạo embedding từ ảnh nếu có
        if file:
            # Tạo đường dẫn lưu tạm thời cho file ảnh
            image_path = os.path.join(TEMP_IMAGE_FOLDER, file.filename)
            file.save(image_path)  # Lưu file ảnh
            try:
                # Tạo embedding từ ảnh
                img_embedding = siamese_model.create_image_embedding(image_path)
            except Exception as e:
                return jsonify({"error": f"Lỗi khi tạo embedding từ ảnh: {str(e)}"}), 500
            finally:
                if os.path.exists(image_path):  # Xóa file ảnh sau khi xử lý
                    os.remove(image_path)

        # Tạo embedding từ tiêu đề nếu có
        if title_query:
            try:
                # Tạo embedding từ tiêu đề
                text_embedding = siamese_model.create_text_embedding(title_query)
            except Exception as e:
                return jsonify({"error": f"Lỗi khi tạo embedding từ tiêu đề: {str(e)}"}), 500

        # Kiểm tra nếu không có embedding nào được tạo
        if img_embedding is None and text_embedding is None:
            return jsonify({"error": "Không thể tạo embedding từ ảnh hoặc tiêu đề"}), 400

        # Truy vấn MongoDB và tìm kiếm
        try:
            # Lấy danh sách sách từ MongoDB
            books = db.books.find({}, {
                "title": 1,
                "price": 1,
                "image_embedding": 1,
                "text_embedding": 1,
                "encoded_image": 1,
                "_id": 0
            })
            # Sử dụng lớp BookSearch để tìm kiếm
            results = book_search.search_books(books, img_embedding, text_embedding)
        except Exception as e:
            return jsonify({"error": f"Lỗi khi truy vấn MongoDB hoặc tìm kiếm: {str(e)}"}), 500

        # Trả về kết quả tìm kiếm
        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": f"Lỗi trong quá trình tìm kiếm: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)