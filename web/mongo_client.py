from pymongo import MongoClient as PyMongoClient  

class CustomMongoClient:
    def __init__(self, uri):
        self.client = PyMongoClient(uri)  
        self.db = self.client['book_search']  

    def get_db(self):
        """Trả về database."""
        return self.db

    def close(self):
        """Đóng kết nối với MongoDB."""
        self.client.close()
