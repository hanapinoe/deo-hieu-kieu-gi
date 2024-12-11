import pandas as pd
import os
from PIL import Image
from dataEmbFunc import encode_image_to_base64, create_text_df_embeddings, create_image_embeddings
from pymongo import MongoClient
from tqdm import tqdm  # Thư viện để hiển thị tiến trình

# Kết nối MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['bookstore']
collection = db['bookCollection']

if __name__ == "__main__":
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv('D:/study/OJT/project/frontend/meta_data.csv')
    dataset = []

    # Duyệt qua từng hàng trong DataFrame
    for _, row in data.iterrows():
        image_path = row['image path']  # Cột chứa đường dẫn ảnh
        title = row['title']            # Cột chứa tiêu đề sách
        price = row['price']            # Cột chứa giá sách

        # Xóa phần 'books/' nếu có trong image_path và thêm vào đường dẫn tĩnh
        if image_path.startswith('books/'):
            image_path = image_path[len('books/'):]
        full_image_path = os.path.join('D:/study/OJT/project/backend/images', image_path)

        # Kiểm tra sự tồn tại của file ảnh
        if not os.path.exists(full_image_path):
            print(f"Image file does not exist: {full_image_path}")  # Thông báo ảnh bị thiếu
            continue  # Bỏ qua nếu file không tồn tại
        
        # Kiểm tra sự tồn tại của file ảnh
        if os.path.exists(full_image_path):
            try:
                # Mở ảnh bằng Pillow
                image = Image.open(full_image_path).convert('RGB')  # Chuyển đổi thành ảnh RGB
                # Thêm thông tin vào dataset
                dataset.append({'image': image, 'title': title, 'price': price, 'image_path': full_image_path})
            except IOError:
                print(f"Cannot open image: {full_image_path}")

    # Tạo embedding văn bản và ảnh
    print("Creating text embeddings...")
    text_embeddings = create_text_df_embeddings(data, column='title')

    print("Creating image embeddings...")
    image_embeddings = create_image_embeddings(dataset)

    # Lưu kết quả vào MongoDB
    print("Saving to MongoDB...")
    for idx, row in tqdm(enumerate(dataset), total=len(dataset), desc="Storing documents"):
        # Mã hóa ảnh thành base64
        encoded_image = encode_image_to_base64(row['image_path'])

        # Tạo document
        document = {
            "title": row['title'],
            "price": row['price'],
            "text_embedding": text_embeddings[idx].tolist(),  # Chuyển sang danh sách để lưu vào MongoDB
            "image_embedding": image_embeddings[idx].tolist(),  # Chuyển sang danh sách
            "encoded_image": encoded_image  # Lưu ảnh mã hóa base64
        }
        
        # Chèn document vào MongoDB
        collection.insert_one(document)

    print("Embedding đã được lưu vào MongoDB thành công.")