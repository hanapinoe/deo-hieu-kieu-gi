from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from paddleocr import PaddleOCR
from pymongo import MongoClient

# Initialize Flask app
app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB connection string
db = client['book_database']
book_collection = db['books']

# Load the trained autoencoder model
class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Load the pretrained model
input_dim = 224 * 224 * 3
model = Autoencoder(input_dim)
model.load_state_dict(torch.load('autoencoder_model.pth'))
model.eval()

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Preprocess image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Extract text from image
def extract_text_from_image(image_path):
    result = ocr.ocr(image_path, cls=True)
    extracted_text = " ".join([line[1][0] for line in result[0]]) if result else ""
    return extracted_text

# Create embeddings
def create_embedding(input_data, input_type='title'):
    if input_type == 'title':
        vectorizer = torch.load('vectorizer.pkl')  # Load the trained vectorizer
        embedding = vectorizer.transform([input_data]).toarray()[0]
    elif input_type == 'image':
        image_tensor = preprocess_image(input_data)
        with torch.no_grad():
            embedding, _ = model(image_tensor.view(-1, input_dim))
        embedding = embedding.numpy()
    return embedding

@app.route('/search', methods=['POST'])
def search_books():
    # Get request data
    image = request.files.get('image')
    title = request.form.get('title')

    if not image and not title:
        return jsonify({"error": "Please provide an image or title."}), 400

    # Process input
    if image:
        image_path = f"temp/{image.filename}"
        image.save(image_path)
        extracted_title = extract_text_from_image(image_path)
        input_embedding = create_embedding(extracted_title, input_type='title')
    elif title:
        input_embedding = create_embedding(title, input_type='title')

    # Retrieve book embeddings from MongoDB
    books = list(book_collection.find())
    book_list = [{"title": book['title'], "embedding": np.array(book['embedding']), "metadata": book['metadata']} for book in books]

    # Calculate cosine similarity
    embeddings = np.array([book['embedding'] for book in book_list])
    similarities = cosine_similarity([input_embedding], embeddings)[0]
    top_matches = sorted(zip(book_list, similarities), key=lambda x: x[1], reverse=True)[:5]

    # Return top 5 results
    response = [{"title": match[0]['title'], "similarity": match[1], "metadata": match[0]['metadata']} for match in top_matches]
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
