from flask import Flask, request, jsonify, render_template, send_from_directory
from pymongo import MongoClient
import os
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Kết nối tới MongoDB Compass
client = MongoClient("mongodb://localhost:27017/")
db = client["bookstore"]  # Tên cơ sở dữ liệu
collection = db["books"]  # Tên collection

# Đường dẫn tới thư mục lưu ảnh
image_folder_path = './craw_image_data/images'
os.makedirs(image_folder_path, exist_ok=True)

# Các cài đặt cho việc tải ảnh lên
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Giới hạn dung lượng file tải lên

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Kiểm tra đuôi file hợp lệ
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# API để tìm kiếm sách và trả về kết quả tương tự
@app.route("/search_books", methods=["POST"])
def search_books():
    book_title = request.form.get("bookTitle")
    image_file = request.files.get("imageFile")

    if not book_title or not image_file or not allowed_file(image_file.filename):
        return jsonify({"success": False, "message": "Vui lòng cung cấp tên sách và ảnh hợp lệ."})

    # Lưu ảnh vào thư mục
    filename = secure_filename(image_file.filename)
    file_path = os.path.join(image_folder_path, filename)
    image_file.save(file_path)

    # Mã hóa ảnh thành Base64
    with open(file_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    # Tìm kiếm sách tương tự trong MongoDB (dựa trên tên sách)
    books = collection.find({"book_title": {"$regex": book_title, "$options": "i"}})

    result_books = []
    for book in books:
        book_info = {
            "book_title": book["book_title"],
            "url": f"/images/{filename}",  # Đưa ảnh ra ngoài để sử dụng trong HTML
            "image_data": f"data:image/jpeg;base64,{image_data}"  # Lưu ảnh vào Base64
        }
        result_books.append(book_info)

    # Trả về kết quả
    if result_books:
        return jsonify({"success": True, "books": result_books})
    else:
        return jsonify({"success": False, "message": "Không tìm thấy sách tương tự."})

# API để phục vụ ảnh từ thư mục 'images' trên localhost
@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory(image_folder_path, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
