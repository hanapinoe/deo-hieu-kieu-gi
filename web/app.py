from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tempfile
import pytesseract
from mongo_client import MongoDBClient
from book_search import BookSearch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Khởi tạo đối tượng MongoDBClient và BookSearch
mongo_client = MongoDBClient()
book_search = BookSearch()

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/search', methods=['POST'])
def search_books():
    try:
        image = request.files.get('image')
        title = request.form.get('title')

        if not image and not title:
            return jsonify({"error": "Vui lòng cung cấp hình ảnh hoặc tiêu đề."}), 400

        if image:
            # Xử lý hình ảnh
            with tempfile.NamedTemporaryFile(delete=False) as temp_image:
                image.save(temp_image.name)
                
                pytesseract.pytesseract.tesseract_cmd = r'c:\users\hon\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\scripts\pytesseract.exe'
                # OCR để lấy văn bản từ hình ảnh
                ocr_text = pytesseract.image_to_string(temp_image.name, lang='vie').strip()

                # Trích xuất embedding ảnh và văn bản
                img_embedding = book_search.create_image_embedding(temp_image.name)
                text_embedding = book_search.create_text_embedding(ocr_text)

        elif title:
            text_embedding = book_search.create_text_embedding(title)
            img_embedding = None

        # Tìm kiếm trong cơ sở dữ liệu
        books = mongo_client.get_books()
        if not books:
            return jsonify({"error": "Không tìm thấy sách trong cơ sở dữ liệu."}), 404

        # Tìm kiếm sách
        results = book_search.search_books(books, img_embedding, text_embedding)

        return jsonify(results)

    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return jsonify({"error": f"Lỗi: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
