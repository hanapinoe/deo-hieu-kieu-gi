import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def create_vectorizer(csv_file_path, output_file_path, max_features=100):
    """
    Tạo và lưu TfidfVectorizer từ tiêu đề sách trong file CSV.

    Args:
        csv_file_path (str): Đường dẫn tới file CSV chứa dữ liệu.
        output_file_path (str): Đường dẫn lưu file vectorizer .pkl.
        max_features (int): Số lượng từ vựng tối đa cho TfidfVectorizer.

    Returns:
        None
    """

    csv_file_path = './books_metadata.csv' 
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

    # Lấy danh sách tiêu đề sách
    titles = books_df['title'].dropna().tolist()  # Loại bỏ các giá trị null

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
csv_file = 'books_data.csv'  # Đường dẫn tới file CSV
output_file = 'vectorizer.pkl'  # Đường dẫn lưu vectorizer
create_vectorizer(csv_file, output_file)
