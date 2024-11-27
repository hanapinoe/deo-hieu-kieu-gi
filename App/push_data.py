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

# Kiểm tra dữ liệu trước khi chèn
valid_data_to_insert = []
for _, row in metadata.iterrows():
    embedding_row = embeddings[embeddings['title'] == row['title']]
    if not embedding_row.empty:
        embedding_row = embedding_row.iloc[0]  # Lấy dòng tương ứng

        # Kiểm tra kích thước embedding
        text_embedding = embedding_row['text_embedding']
        image_embedding = embedding_row['image_embedding']

        if len(text_embedding) != 101:
            print(f"Bỏ qua sách '{row['title']}' vì kích thước text_embedding không hợp lệ: {len(text_embedding)}")
            continue
        if len(image_embedding) != 2048:
            print(f"Bỏ qua sách '{row['title']}' vì kích thước image_embedding không hợp lệ: {len(image_embedding)}")
            continue

        # Tạo document
        document = {
            "title": row['title'],
            "price": float(row['price']),  # Chuyển sang float để tránh lỗi serialize
            "image_url": row['image_url'],
            "text_embedding": [float(x) for x in text_embedding],  # Đảm bảo là list of floats
            "image_embedding": [float(x) for x in image_embedding]
        }
        valid_data_to_insert.append(document)

# Chèn dữ liệu hợp lệ vào MongoDB
if valid_data_to_insert:
    collection.insert_many(valid_data_to_insert)
    print(f"Đã chèn thành công {len(valid_data_to_insert)} bản ghi vào MongoDB.")
else:
    print("Không có bản ghi hợp lệ để chèn vào MongoDB. Kiểm tra dữ liệu của bạn.")
