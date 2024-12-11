from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BookSearch:
    def __init__ (self, books, img_embedding=None, text_embedding=None):
        self.books = books
        self.img_embedding = img_embedding
        self.text_embedding = text_embedding

    def search_books(self):
        """Tìm kiếm sách dựa trên hình ảnh và văn bản."""
        results = []

        for book in self.books:
            stored_img_embedding = np.array(book['image_embedding']).reshape(1, -1)
            stored_text_embedding = np.array(book['text_embedding']).reshape(1, -1)

            img_similarity = cosine_similarity(self.img_embedding, stored_img_embedding)[0][0] if self.img_embedding is not None else 0
            text_similarity = cosine_similarity(self.text_embedding, stored_text_embedding)[0][0] if self.text_embedding is not None else 0

            overall_similarity = (img_similarity + text_similarity) / (2 if self.img_embedding is not None and self.text_embedding is not None else 1)

            results.append({
                "title": book["title"],
                "price": book["price"],
                "relevance_score": float(overall_similarity),
                "encoded_image": book["encoded_image"]
            })

        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:10]