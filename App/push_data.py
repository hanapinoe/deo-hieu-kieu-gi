import pandas as pd
from pymongo import MongoClient
import json
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
embeddings['text_embedding'] = embeddings['text_embedding'].apply(json.loads)
embeddings['image_embedding'] = embeddings['image_embedding'].apply(json.loads)

# Tạo URL hình ảnh từ đường dẫn ảnh trong metadata
metadata['image_url'] = metadata['title'].apply(lambda x: os.path.join('./static/images', x.replace(' ', '_').lower() + '.jpg'))

# Tổng hợp dữ liệu từ metadata và embeddings
data_to_insert = []
for _, row in metadata.iterrows():
    embedding_row = embeddings[embeddings['title'] == row['title']].iloc[0]
    document = {
        "title": row['title'],
        "price": row['price'],
        "image_url": row['image_url'],
        "text_embedding": embedding_row['text_embedding'],  # Đã chuyển đổi thành list
        "image_embedding": embedding_row['image_embedding']  # Đã chuyển đổi thành list
    }
    data_to_insert.append(document)

# Chèn dữ liệu vào MongoDB
collection.insert_many(data_to_insert)
print(f"Successfully inserted {len(data_to_insert)} records into the book_embeddings collection in the book_db database.")
