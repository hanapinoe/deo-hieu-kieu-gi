import requests
from bs4 import BeautifulSoup
import os
import csv

# URL cơ bản của trang web mà bạn muốn cào dữ liệu
base_url = 'https://savina.com.vn/tat-ca-san-pham?page='

# Tạo thư mục lưu ảnh
if not os.path.exists('images'):
    os.makedirs('images')

# Tạo  tệp CSV để lưu metadata
csv_file = 'books_data.csv'
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Ghi tiêu đề cột
    writer.writerow(['Image Path', 'Book Title','Price'])

# Hàm để cào dữ liệu từ một trang
def scrape_page(url, page_num):
    response = requests.get(url + str(page_num))
    
    if response.status_code != 200:
        print(f"Failed to fetch page {page_num}. Status code: {response.status_code}")
        return False

    soup = BeautifulSoup(response.text, 'html.parser')

    # Tìm tất cả các phần tử sản phẩm
    products = soup.find_all('li', class_='item_product col-md-3 col-sm-6 col-xs-6 col-item-all')

    if not products:
        return False  # Dừng nếu không tìm thấy sản phẩm

    # Duyệt qua từng sản phẩm để lấy thông tin
    for i, product in enumerate(products):
        # 1. Lấy thông tin ảnh
        img_path = 'No Image'  # Gán giá trị mặc định cho img_path
        img_url = None
        img_tag = product.find('picture')
        if img_tag:
            source_tags = img_tag.find_all('source')
            max_resolution = -1
            for source in source_tags:
                srcset = source.get('srcset', '')
                if srcset:
                    srcset_items = srcset.split(',')
                    for item in srcset_items:
                        item = item.strip()
                        try:
                            url, resolution = item.split(' ')
                            resolution = int(resolution.replace('w', '').strip())
                            if resolution > max_resolution:
                                img_url = url
                                max_resolution = resolution
                        except ValueError:
                            img_url = item.strip()
                            max_resolution = float('inf')
        if img_url:
            if not img_url.startswith('http'):
                img_url = 'https:' + img_url
            img_data = requests.get(img_url).content
            img_path = f'images/image_page{page_num}_{i+1}.jpg'
            with open(img_path, 'wb') as f:
                f.write(img_data)
            print(f"Downloaded image: {img_path}")

        # 2. Lấy tiêu đề sách
        title_tag = product.find('h3')
        title = title_tag.text.strip() if title_tag else 'Unknown Title'

        # 3. Lấy giá sản phẩm (bỏ qua giá gạch bỏ)
        price_tag = product.find('p', class_='pro-price')  # Tìm thẻ chứa giá
        if price_tag:
            # Loại bỏ phần tử `<del>` nếu có
            for del_tag in price_tag.find_all('del'):
                del_tag.extract()  # Xóa thẻ `<del>` khỏi nội dung của `price_tag`
            current_price = price_tag.get_text(strip=True)  # Lấy giá hiện tại sau khi loại bỏ thẻ `<del>`
        else:
            current_price = 'Unknown Price'

        # Lưu thông tin vào tệp CSV
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([img_path, title, current_price])

    return True



#Cào dữ liệu từ trang đầu tiên và tiếp tục cho đến khi không còn trang nào nữa
page_num = 1
while True:
    print(f"Scraping page {page_num}...")
    if not scrape_page(base_url, page_num):
        print(f"No more data found at page {page_num}. Stopping.")
        break
    page_num += 1

print("Scraping completed. Data saved to 'books_data.csv' and images folder.")
