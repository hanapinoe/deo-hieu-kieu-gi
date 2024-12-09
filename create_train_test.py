import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc file CSV đầu vào
input_csv = 'meta_data.csv'  # Thay bằng tên file của bạn
output_train_file = 'train_pairs.npy'
output_test_file = 'test_pairs.npy'
output_train_labels_file = 'train_labels.npy'
output_test_labels_file = 'test_labels.npy'

# Đọc dữ liệu
df = pd.read_csv(input_csv)
print(f"Đã tải CSV với các cột: {df.columns.tolist()}")

# Kiểm tra cột 'image path'
if 'image path' not in df.columns or 'title' not in df.columns:
    raise ValueError("'image path' hoặc 'title' không có trong file CSV.")

# Hàm tạo embeddings cho hình ảnh (thay bằng logic thực tế của bạn)
def create_image_embedding(image_path):
    # Placeholder: Trả về vector embedding có kích thước 2048 (đúng với mô hình ResNet-50)
    return np.random.rand(2048)

# Hàm tạo embeddings cho văn bản (thay bằng logic thực tế của bạn)
def create_text_embedding(title):
    # Placeholder: Trả về vector embedding có kích thước 128
    return np.random.rand(128)

# Chuẩn bị các cặp hình ảnh và văn bản
pairs = []
labels = []

for index, row in df.iterrows():
    try:
        # Đảm bảo đường dẫn hình ảnh tồn tại
        image_path = row['image path']
        if not os.path.exists(image_path):
            print(f"Ảnh không tìm thấy: {image_path}. Bỏ qua dòng {index}.")
            continue
        
        # Tạo embeddings cho hình ảnh và văn bản
        img_embedding = create_image_embedding(image_path)
        text_embedding = create_text_embedding(row['title'])
        
        # Tạo cặp đúng (label = 1)
        pairs.append([img_embedding, text_embedding])
        labels.append(1)
        
        # Tạo cặp ngẫu nhiên không khớp (label = 0)
        random_row = df.sample(1).iloc[0]
        random_text_embedding = create_text_embedding(random_row['title'])
        pairs.append([img_embedding, random_text_embedding])
        labels.append(0)
        
    except Exception as e:
        print(f"Lỗi khi xử lý dòng {index}: {e}")

# Đảm bảo dữ liệu không rỗng
if len(pairs) == 0 or len(labels) == 0:
    raise ValueError("Không có dữ liệu hợp lệ. Kiểm tra lại dataset và đường dẫn.")

# Chia dữ liệu thành train và test
train_pairs, test_pairs, train_labels, test_labels = train_test_split(
    pairs, labels, test_size=0.2, random_state=42
)

# Lưu cặp train và test vào file .npy
np.save(output_train_file, np.array(train_pairs, dtype=object))  # Sử dụng dtype=object
np.save(output_test_file, np.array(test_pairs, dtype=object))    # Sử dụng dtype=object
np.save(output_train_labels_file, train_labels)
np.save(output_test_labels_file, test_labels)

print("Chia dữ liệu thành công.")
print(f"Số lượng cặp train: {len(train_pairs)}, Số lượng cặp test: {len(test_pairs)}")
