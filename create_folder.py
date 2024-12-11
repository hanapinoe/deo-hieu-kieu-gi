import os
import shutil

# Đường dẫn tới các thư mục
train_folder = 'train_images'
test_folder = 'test_images'  # Đổi tên cho đúng với thư mục thực tế của bạn

# Tạo thư mục con cho lớp (ví dụ: 'class_1') trong train_images và test_images
def create_class_folders(folder_path, class_name):
    # Tạo thư mục lớp nếu chưa có
    class_folder = os.path.join(folder_path, class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    
    # Di chuyển tất cả hình ảnh vào thư mục lớp
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        
        # Kiểm tra chỉ di chuyển các file ảnh (.jpg, .png, .jpeg)
        if os.path.isfile(img_path) and (img_file.endswith('.jpg') or img_file.endswith('.png') or img_file.endswith('.jpeg')):
            shutil.move(img_path, os.path.join(class_folder, img_file))

# Tạo thư mục lớp cho train_images và test_images
create_class_folders(train_folder, 'class_1')
create_class_folders(test_folder, 'class_1')

print("Đã tạo thư mục lớp và phân loại hình ảnh vào các lớp.")
