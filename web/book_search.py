import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import joblib

class BookSearch:
    def __init__(self):
        # Tải vectorizer và mô hình ResNet-50
        self.vectorizer = joblib.load('D:/Project/web/vectorizer.pkl')
        self.resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_model = torch.nn.Sequential(*list(self.resnet_model.children())[:-1])

    def preprocess_title(self, title):
        """Chuẩn hóa tiêu đề người dùng nhập."""
        import re
        title = title.lower().strip()
        title = re.sub(r'[^\w\s]', '', title)
        return title

    def create_text_embedding(self, title):
        """Tạo embedding từ tiêu đề văn bản sau khi chuẩn hóa."""
        title = self.preprocess_title(title)
        text_embedding = self.vectorizer.transform([title]).toarray()
        text_embedding = torch.tensor(text_embedding, dtype=torch.float)
        transform_layer = torch.nn.Linear(text_embedding.size(1), 128)

        with torch.no_grad():
            text_embedding = transform_layer(text_embedding).detach().numpy()

        return np.nan_to_num(text_embedding).reshape(1, -1)

    def preprocess_image(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)

    def create_image_embedding(self, image_path):
        image_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            features = self.resnet_model(image_tensor)
        img_embedding = features.view(features.size(0), -1).cpu().numpy()
        return np.nan_to_num(img_embedding).reshape(1, -1)

    def search_books(self, books, img_embedding=None, text_embedding=None):
        results = []
        for book in books:
            stored_img_embedding = np.array(book['image_embedding']).reshape(1, -1)
            stored_text_embedding = np.array(book['text_embedding']).reshape(1, -1)

            img_similarity = cosine_similarity(img_embedding, stored_img_embedding)[0][0] if img_embedding is not None else 0
            text_similarity = cosine_similarity(text_embedding, stored_text_embedding)[0][0]

            overall_similarity = (img_similarity + text_similarity) / (2 if img_embedding is not None else 1)

            results.append({
                "title": book["title"],
                "price": book["price"],
                "relevance_score": float(overall_similarity),
                "encoded_image": book["encoded_image"]
            })

        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:5]
