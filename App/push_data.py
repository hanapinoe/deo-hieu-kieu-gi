import pandas as pd
from pymongo import MongoClient
import ast
import os
import base64

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
output_image_dir = './static/images'
os.makedirs(output_image_dir, exist_ok=True)

metadata['image_url'] = metadata['title'].apply(
    lambda x: os.path.join(output_image_dir, x.replace(' ', '_').lower() + '.jpg')
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
        
        # Kiểm tra kích thước text_embedding
        if len(text_embedding) != 101:  # Mô hình yêu cầu text_embedding kích thước 101
            print(f"Skipping title '{row['title']}' due to incorrect text_embedding size: {len(text_embedding)}")
            if len(text_embedding) < 101:
                text_embedding = text_embedding + [0] * (101 - len(text_embedding))  # Padding bằng 0
            elif len(text_embedding) > 101:
                text_embedding = text_embedding[:101]  # Cắt bớt nếu quá dài
            else:
                continue
        
        # Kiểm tra kích thước image_embedding
        if len(image_embedding) != 2048:  # ResNet50 yêu cầu image_embedding kích thước 2048
            print(f"Skipping title '{row['title']}' due to incorrect image_embedding size: {len(image_embedding)}")
            if len(image_embedding) < 2048:
                image_embedding = image_embedding + [0] * (2048 - len(image_embedding))  # Padding
            elif len(image_embedding) > 2048:
                image_embedding = image_embedding[:2048]  # Cắt bớt

        # Lưu hình ảnh encode_img
import os
import base64

# Đảm bảo thư mục tồn tại trước khi lưu ảnh
image_dir = './static/images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Lưu hình ảnh encode_img (base64 -> ảnh thực)
image_path = row['image_url']  # Lấy image_url từ row
encode_img = embedding_row.get('encode_img', None)  # Lấy encode_img từ embedding

if encode_img:
    try:
        # Xử lý lưu ảnh từ base64
        image_filename = os.path.basename(image_path)  # Lấy tên ảnh từ đường dẫn
        saved_image_path = os.path.join(image_dir, image_filename)
        
        # Decode base64 và lưu vào file
        with open(saved_image_path, "wb") as image_file:
            image_file.write(base64.b64decode(encode_img))
        
        # Mã hóa ảnh vừa lưu lại thành base64 để lưu vào MongoDB
        with open(saved_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Tạo document hợp lệ để chèn vào MongoDB
        document = {
            "title": row['title'],
            "price": row['price'],
            "text_embedding": text_embedding,
            "image_embedding": image_embedding,
            "encoded_image": encoded_image  # Lưu ảnh dưới dạng base64
        }
        valid_data_to_insert.append(document)

    except Exception as e:
        print(f"Failed to save or encode image for title '{row['title']}': {e}")


# Chèn dữ liệu hợp lệ vào MongoDB
if valid_data_to_insert:
    collection.insert_many(valid_data_to_insert)
    print(f"Successfully inserted {len(valid_data_to_insert)} records into the books collection in the bookstore database.")
else:
    print("No valid records to insert. Please check the data.")
