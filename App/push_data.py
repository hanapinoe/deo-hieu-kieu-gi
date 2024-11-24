import os
import pymongo
import csv
import base64

# Kết nối tới MongoDB Compass (localhost:27017)
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["bookstore"]  # Tên cơ sở dữ liệu
collection = db["books"]  # Tên collection

# Đường dẫn tới file CSV và thư mục chứa ảnh
csv_file_path = "./craw_image_data/books_data.csv"  # File CSV chứa tên sách và đường dẫn ảnh
image_folder_path = "./craw_image_data/images"  # Thư mục chứa ảnh

# Kiểm tra xem thư mục ảnh có tồn tại hay không
if not os.path.exists(image_folder_path):
    print(f"Không tìm thấy thư mục ảnh: {image_folder_path}")
    exit()

# Danh sách để lưu các tài liệu sẽ chèn vào MongoDB
documents = []

# Đọc dữ liệu từ file CSV và xử lý ảnh
if not os.path.exists(csv_file_path):
    print(f"Không tìm thấy file CSV: {csv_file_path}")
    exit()

# Đọc file CSV
with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file)  # Đọc dữ liệu dạng list
    for row in reader:
        if len(row) == 2:  # Kiểm tra nếu dòng có 2 phần: đường dẫn ảnh và tên sách
            image_path = row[0].strip()  # Đường dẫn ảnh từ CSV
            book_title = row[1].strip()  # Tên sách từ CSV
            
            # Kiểm tra nếu file ảnh có tồn tại trong thư mục
            image_name = os.path.basename(image_path)  # Lấy tên ảnh
            full_image_path = os.path.join(image_folder_path, image_name)  # Đường dẫn đầy đủ ảnh
            
            if os.path.exists(full_image_path):
                # Chuyển đổi ảnh thành Base64
                with open(full_image_path, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode("utf-8")

                # Tạo URL cho ảnh (URL trên localhost sẽ được xây dựng như sau)
                url = f"http://localhost:5000/images/{image_name}"  # URL ảnh trên localhost

                # Tạo document cho MongoDB
                document = {
                    "book_title": book_title,  # Tên sách
                    "image_path": image_path,  # Đường dẫn ảnh trong CSV
                    "url": url,  # URL ảnh trên localhost
                    "image_data": f"data:image/jpeg;base64,{image_data}"  # Mã hóa ảnh thành Base64
                }
                documents.append(document)
            else:
                print(f"Không tìm thấy ảnh với tên: {image_name}")

# Chèn dữ liệu vào MongoDB
if documents:
    result = collection.insert_many(documents)
    print(f"Đã chèn {len(result.inserted_ids)} tài liệu vào MongoDB.")
else:
    print("Không có tài liệu nào để chèn.")
