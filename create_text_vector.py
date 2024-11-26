import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Đọc dữ liệu từ file CSV
books_df = pd.read_csv('books_data.csv')  # Đảm bảo file CSV có cột 'title'

# Lấy danh sách tiêu đề sách từ cột 'title'
titles = books_df['title'].tolist()

# Tạo TfidfVectorizer và huấn luyện nó với dữ liệu tiêu đề sách
vectorizer = TfidfVectorizer(max_features=100)  # Giới hạn số lượng từ vựng nếu cần
X = vectorizer.fit_transform(titles)

# Lưu vectorizer vào file .pkl
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Vectorizer đã được lưu thành công vào 'vectorizer.pkl'")
