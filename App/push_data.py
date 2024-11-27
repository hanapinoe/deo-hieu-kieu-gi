import pandas as pd
from pymongo import MongoClient
import json

# Kết nối tới MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Thay thế bằng URL của MongoDB nếu cần
db = client['bookstore']
collection = db['books']

# Đọc dữ liệu từ file CSV
df = pd.read_csv('embeddings_for_mongodb.csv')

# Chuyển đổi dữ liệu từ DataFrame thành danh sách các từ điển
data_to_insert = []
for _, row in df.iterrows():
    document = {
        "title": row['title'],
        "price": row['price'],
        "image_url": row['image_url'],
        "text_embedding": json.loads(row['text_embedding']),  # Chuyển đổi chuỗi JSON thành list
        "image_embedding": json.loads(row['image_embedding'])  # Chuyển đổi chuỗi JSON thành list
    }
    data_to_insert.append(document)

# Chèn dữ liệu vào MongoDB
collection.insert_many(data_to_insert)
print(f"Successfully inserted {len(data_to_insert)} records into the book_embeddings collection in the book_db database.")
