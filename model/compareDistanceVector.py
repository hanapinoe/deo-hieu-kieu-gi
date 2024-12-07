from imgInputEmb import imgInputEmb
from pymongo import MongoClient

client = MongoClient("URL") #Ket noi den mongodb

db = client["DBs"] #Truy cap co so du lieu

collection = db["my_collection"] #truy cap mot collection










client.close()


