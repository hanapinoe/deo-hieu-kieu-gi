import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import joblib

class SiameseModel:
    def __init__(self, model_path=None, vectorizer_path=None):
        self.vectorizer = joblib.load(vectorizer_path) if vectorizer_path else None
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  

        self.img_transform = torch.nn.Linear(2048, 128)  
        self.text_transform = torch.nn.Linear(128, 128)

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Tải mô hình Siamese từ file, bỏ qua lớp không tương thích."""
        try:
            self.model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)  
            print("Mô hình đã được tải thành công!")
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")

    def preprocess_image(self, image_path):
        """Tiền xử lý hình ảnh đầu vào cho mô hình."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)

    def create_image_embedding(self, image_path):
        """Tạo embedding từ hình ảnh."""
        image_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            features = self.model(image_tensor)
        img_embedding = features.view(features.size(0), -1).cpu().numpy()
        img_embedding = np.nan_to_num(img_embedding).reshape(1, -1)

        img_embedding = torch.tensor(img_embedding, dtype=torch.float)
        with torch.no_grad():
            img_embedding = self.img_transform(img_embedding).detach().numpy()

        return img_embedding

    def preprocess_title(self, title):
        """Chuẩn hóa tiêu đề văn bản."""
        import re
        title = title.lower().strip()
        title = re.sub(r'[^\w\s]', '', title)
        return title

    def create_text_embedding(self, title):
        """Tạo embedding cho tiêu đề văn bản."""
        title = self.preprocess_title(title)
        text_embedding = self.vectorizer.transform([title]).toarray()
        text_embedding = torch.tensor(text_embedding, dtype=torch.float)

        with torch.no_grad():
            text_embedding = self.text_transform(text_embedding).detach().numpy()

        return np.nan_to_num(text_embedding).reshape(1, -1)