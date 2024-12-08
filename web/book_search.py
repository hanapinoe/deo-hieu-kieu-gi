from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BookSearch:
    def __init__(self, model):
        """Khởi tạo đối tượng tìm kiếm sách với mô hình Siamese."""
        self.siamese_model = model

    def search_books(self, books, img_embedding=None, text_embedding=None):
        """Tìm kiếm sách dựa trên hình ảnh và văn bản."""
        results = []

        for book in books:
            # Lấy embedding từ cơ sở dữ liệu
            stored_img_embedding = np.array(book['image_embedding']).reshape(1, -1)
            stored_text_embedding = np.array(book['text_embedding']).reshape(1, -1)

            # Tính độ tương đồng cosine
            img_similarity = cosine_similarity(img_embedding, stored_img_embedding)[0][0] if img_embedding is not None else 0
            text_similarity = cosine_similarity(text_embedding, stored_text_embedding)[0][0] if text_embedding is not None else 0

            # Tính độ tương đồng tổng hợp
            overall_similarity = (img_similarity + text_similarity) / (2 if img_embedding is not None and text_embedding is not None else 1)

            # Thêm sách và điểm tương đồng vào danh sách kết quả
            results.append({
                "title": book["title"],
                "price": book["price"],
                "relevance_score": float(overall_similarity),
                "encoded_image": book["encoded_image"]
            })

        # Sắp xếp kết quả theo độ tương đồng giảm dần và lấy 10 sách đầu tiên
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:10]