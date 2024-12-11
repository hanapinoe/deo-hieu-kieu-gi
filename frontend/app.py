from flask import Flask, request, jsonify, render_template
from PIL import Image
import os
import numpy as np
from backend.mongo_client import CustomMongoClient
from backend.dataEmbFunc import encode_image_to_base64, create_text_embeddings, create_image_embeddings
from backend.book_search import BookSearch

app = Flask(__name__)

MODEL_PATH = './model/siamese_model.pth' 
VECTORIZER_PATH = './model/vectorizer.pkl'

TEMP_IMAGE_FOLDER = 'temp_images'
if not os.path.exists(TEMP_IMAGE_FOLDER):
    os.makedirs(TEMP_IMAGE_FOLDER)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

try:
    mongo_client = CustomMongoClient(os.getenv('MONGO_URL'))
    db = mongo_client.get_db()
    print("Kết nối MongoDB thành công!")
except Exception as e:
    print(f"Lỗi khi kết nối MongoDB: {e}")

@app.route('/')
def home():
    """Trang chủ - Giao diện frontend."""
    return render_template('frontend1.html')

img_embedding = None
text_embedding = None

@app.route('/reset', methods=['POST'])
def reset_state():
    """Reset trạng thái của server, xóa embedding và dữ liệu tạm."""
    global img_embedding, text_embedding

    try:
        img_embedding = None
        text_embedding = None

        for file_name in os.listdir(TEMP_IMAGE_FOLDER):
            file_path = os.path.join(TEMP_IMAGE_FOLDER, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        print("Đã reset trạng thái embedding và xóa dữ liệu tạm.")
        return jsonify({"message": "Trạng thái đã được reset thành công."}), 200
    except Exception as e:
        return jsonify({"error": f"Lỗi khi reset trạng thái: {str(e)}"}), 500


@app.route('/search', methods=['POST'])
def search_books():
    global img_embedding, text_embedding

    try:
        data = request.form
        file = request.files.get('image_path')  
        title_query = data.get('title', None)  

        if not file and not title_query:
            return jsonify({"error": "Cần cung cấp ít nhất một trong hai: 'image_path' hoặc 'title'"}), 400

        if file:
            image_path = os.path.join(TEMP_IMAGE_FOLDER, file.filename)  # type: ignore
            try:
                file.save(image_path)
                img_embedding = create_image_embeddings(image_path)
            except Exception as e:
                return jsonify({"error": f"Lỗi khi tạo embedding từ ảnh: {str(e)}"}), 500
            finally:
                if os.path.exists(image_path):  
                    os.remove(image_path)

        if title_query:
            try:
                text_embedding = create_text_embeddings(title_query)
            except Exception as e:
                return jsonify({"error": f"Lỗi khi tạo embedding từ tiêu đề: {str(e)}"}), 500
            
        if img_embedding is None and text_embedding is None:
            return jsonify({"error": "Không thể tạo embedding từ ảnh hoặc tiêu đề"}), 400

        try:
            books = db.books.find({}, {
                "title": 1,
                "price": 1,
                "image_embedding": 1,
                "text_embedding": 1,
                "encoded_image": 1,
                "_id": 0
            })
            results = BookSearch.search_books(books, img_embedding, text_embedding) # type: ignore
        except Exception as e:
            return jsonify({"error": f"Lỗi khi truy vấn MongoDB hoặc tìm kiếm: {str(e)}"}), 500

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": f"Lỗi trong quá trình tìm kiếm: {str(e)}"}), 500
    
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Dung lượng file tải lên vượt quá giới hạn 16MB!"}), 413


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)