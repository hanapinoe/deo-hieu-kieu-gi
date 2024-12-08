from pymongo import MongoClient

class MongoDBClient:
    def __init__(self, uri='mongodb://localhost:27017/', db_name='book_search', collection_name='books'):
        try:
            self.client = MongoClient(uri)
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            print("Kết nối với MongoDB thành công.")
        except Exception as e:
            print(f"Không thể kết nối với MongoDB: {str(e)}")
            exit(1)

    def get_books(self):
        return list(self.collection.find())

    def save_book(self, book_data):
        self.collection.insert_one(book_data)
