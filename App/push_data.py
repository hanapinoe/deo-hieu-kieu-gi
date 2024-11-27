import pandas as pd
from pymongo import MongoClient
import json
import ast
import os

# Kết nối tới MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Thay thế bằng URL của MongoDB nếu cần
db = client['bookstore']
collection = db['books']

# Đọc dữ liệu từ file metadata.csv
metadata = pd.read_csv('books_metadata.csv')

# Đọc dữ liệu từ file embeddings_for_mongodb.csv
embeddings = pd.read_csv('embeddings_for_mongodb.csv')

# Chuyển đổi các cột embedding từ chuỗi JSON sang list
embeddings['text_embedding'] = embeddings['text_embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
embeddings['image_embedding'] = embeddings['image_embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Kiểm tra kích thước của các embeddings
for i, row in embeddings.iterrows():
    if len(row['text_embedding']) != 100:
        print(f"Warning: Text embedding at index {i} has incorrect length: {len(row['text_embedding'])}")
    if len(row['image_embedding']) != 1000:
        print(f"Warning: Image embedding at index {i} has incorrect length: {len(row['image_embedding'])}")

# Tạo URL hình ảnh từ đường dẫn ảnh trong metadata
metadata['image_url'] = metadata['title'].apply(lambda x: os.path.join('./static/images', x.replace(' ', '_').lower() + '.jpg'))

# Đảm bảo embedding có kích thước phù hợp với mô hình
valid_data_to_insert = []
for _, row in metadata.iterrows():
    embedding_row = embeddings[embeddings['title'] == row['title']]
    if not embedding_row.empty:
        embedding_row = embedding_row.iloc[0]
        
        # Kiểm tra và sửa kích thước embedding
        text_embedding = embedding_row['text_embedding']
        image_embedding = embedding_row['image_embedding']
        
        if len(text_embedding) != 101:  # Mô hình của bạn yêu cầu text_embedding kích thước 101
            print(f"Skipping title '{row['title']}' due to incorrect text_embedding size: {len(text_embedding)}")
            continue
        if len(image_embedding) != 2048:  # ResNet50 yêu cầu image_embedding kích thước 2048
            print(f"Skipping title '{row['title']}' due to incorrect image_embedding size: {len(image_embedding)}")
            continue

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

