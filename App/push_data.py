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
metadata = pd.read_csv('metadata.csv')

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

# Tổng hợp dữ liệu từ metadata và embeddings
data_to_insert = []
for _, row in metadata.iterrows():
    embedding_row = embeddings[embeddings['title'] == row['title']]
    if not embedding_row.empty:
        embedding_row = embedding_row.iloc[0]
        document = {
            "title": row['title'],
            "price": row['price'],
            "image_url": row['image_url'],
            "text_embedding": embedding_row['text_embedding'],  # Đã chuyển đổi thành list
            "image_embedding": embedding_row['image_embedding']  # Đã chuyển đổi thành list
        }
        data_to_insert.append(document)

# Chèn dữ liệu vào MongoDB
if data_to_insert:
    collection.insert_many(data_to_insert)
    print(f"Successfully inserted {len(data_to_insert)} records into the books collection in the bookstore database.")
else:
    print("No records to insert. Please check the data.")
