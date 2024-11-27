import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def create_vectorizer(csv_file_path, output_file_path, max_features=100):
    

    # Kiểm tra file CSV
    if not os.path.exists(csv_file_path):
        print(f"Không tìm thấy file CSV: {csv_file_path}")
        return

    # Đọc dữ liệu từ file CSV
    books_df = pd.read_csv(csv_file_path)

    # Kiểm tra cột 'title' có tồn tại không
    if 'title' not in books_df.columns:
        print("File CSV không có cột 'title'.")
        return

    # Lấy danh sách tiêu đề sách và loại bỏ các giá trị null
    titles = books_df['title'].dropna().tolist()

    if not titles:
        print("Không có dữ liệu tiêu đề sách trong file CSV.")
        return

    # Tạo TfidfVectorizer và huấn luyện
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(titles)

    # Lưu vectorizer vào file .pkl
    joblib.dump(vectorizer, output_file_path)
    print(f"Vectorizer đã được lưu thành công vào '{output_file_path}'")

# Sử dụng hàm để tạo vectorizer
csv_file = 'books_metadata.csv'  # Đảm bảo đường dẫn đúng tới file CSV
output_file = 'vectorizer.pkl'  # Đảm bảo đường dẫn đúng để lưu file .pkl
create_vectorizer(csv_file, output_file)
