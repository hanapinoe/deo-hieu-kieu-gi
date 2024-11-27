import pandas as pd
from pymongo import MongoClient
import ast
import os

# Kết nối tới MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Thay URL nếu cần
db = client['bookstore']
collection = db['books']

# Đọc dữ liệu metadata
metadata_file = 'books_metadata.csv'
embeddings_file = 'embeddings_for_mongodb.csv'

# Kiểm tra file có tồn tại không
if not os.path.exists(metadata_file) or not os.path.exists(embeddings_file):
    raise FileNotFoundError("Không tìm thấy file 'books_metadata.csv' hoặc 'embeddings_for_mongodb.csv'")

# Đọc file
metadata = pd.read_csv(metadata_file)
embeddings = pd.read_csv(embeddings_file)

# Chuyển đổi embedding từ chuỗi JSON sang list
embeddings['text_embedding'] = embeddings['text_embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
embeddings['image_embedding'] = embeddings['image_embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Tạo đường dẫn ảnh (nếu cần thay đổi đường dẫn gốc, sửa tại đây)
metadata['image_url'] = metadata['title'].apply(
    lambda x: os.path.join('./static/images', x.replace(' ', '_').lower() + '.jpg')
)

# Kiểm tra và sửa kích thước embedding
valid_data_to_insert = []
for _, row in metadata.iterrows():
    embedding_row = embeddings[embeddings['title'] == row['title']]
    if not embedding_row.empty:
        embedding_row = embedding_row.iloc[0]
        
        # Kiểm tra và sửa kích thước embedding
        text_embedding = embedding_row['text_embedding']
        image_embedding = embedding_row['image_embedding']
        
        # Kiểm tra nếu text_embedding có kích thước 101
        if len(text_embedding) != 101:  # Mô hình của bạn yêu cầu text_embedding kích thước 101
            print(f"Skipping title '{row['title']}' due to incorrect text_embedding size: {len(text_embedding)}")
            continue
        
        # Kiểm tra nếu image_embedding có kích thước 2048
        if len(image_embedding) != 2048:  # ResNet50 yêu cầu image_embedding kích thước 2048
            print(f"Skipping title '{row['title']}' due to incorrect image_embedding size: {len(image_embedding)}")
            continue

        # Tạo document hợp lệ để chèn vào MongoDB
        document = {
            "title": row['title'],
            "price": row['price'],
            "image_url": row['image_url'],
            "text_embedding": text_embedding,
            "image_embedding": image_embedding
        }
        valid_data_to_insert.append(document)

# Chèn dữ liệu hợp lệ vào MongoDB
if valid_data_to_insert:
    collection.insert_many(valid_data_to_insert)
    print(f"Successfully inserted {len(valid_data_to_insert)} records into the books collection in the bookstore database.")
else:
    print("No valid records to insert. Please check the data.")

