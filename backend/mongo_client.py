from pymongo import MongoClient as PyMongoClient  
import os

class CustomMongoClient:
    def __init__(self, uri=None):
        uri = uri or os.getenv('MONGO_URI')
        if not uri:
            raise ValueError("Biến môi trường 'MONGO_URI' chưa được cấu hình!")
        self.client = PyMongoClient(uri)  
        self.db = self.client['book_search'] 

    def get_db(self):
        """Trả về database."""
        return self.db

    def get_collection(self, collection_name):
        """Trả về collection theo tên."""
        return self.db[collection_name]

    def close(self):
        """Đóng kết nối với MongoDB."""
        self.client.close()
