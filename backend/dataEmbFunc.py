import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import base64
from sentence_transformers import SentenceTransformer
from torchvision.models import ResNet50_Weights


def create_text_embeddings(text):
    modelText = SentenceTransformer("C:/Users/dodof/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/9a27583f9c2cc7c03a95c08c5f087318109e2613")
    if not text or not isinstance(text, str):
        raise ValueError("Invalid text input for embedding.")
    return modelText.encode(text)

# Cập nhật hàm tạo embedding văn bản
def create_text_df_embeddings(df, column: str):
    modelText = SentenceTransformer("C:/Users/dodof/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/9a27583f9c2cc7c03a95c08c5f087318109e2613")
    if df is None or column not in df:
        raise ValueError(f"Invalid DataFrame or column: {column}")

    # Generate embeddings for text in the specified column
    return modelText.encode(df[column].tolist())
    

# Hàm tạo embedding ảnh
def create_image_embeddings(data):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Loại bỏ lớp cuối (fully connected layer)
    model = torch.nn.Sequential(*list(model.children())[:-1])

    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_image_embedding(image):
        image = preprocess(image).unsqueeze(0)  # Thêm batch dimension
        with torch.no_grad():
            features = model(image)  # Đầu ra của ResNet50 sau lớp pooling
        return features.view(-1).numpy()  # Flatten vector từ (1, 2048) thành (2048,)

    return np.array([get_image_embedding(d['image']) for d in data])


# Hàm mã hóa ảnh thành base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image

