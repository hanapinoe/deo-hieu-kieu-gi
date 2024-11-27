from flask import Flask, request, jsonify
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["book_database"]
collection = db["books"]

vectorizer = joblib.load('vectorizer.pkl')

@app.route('/search', methods=['POST'])
def search_books():
    data = request.json
    title = data.get("title")
    input_embedding = vectorizer.transform([title]).toarray()[0].tolist()

    books = list(collection.find({}, {"_id": 0, "title": 1, "price": 1, "text_embedding": 1}))
    results = []

    for book in books:
        similarity = cosine_similarity([input_embedding], [book["text_embedding"]])[0][0]
        results.append({
            "title": book["title"],
            "price": book["price"],
            "similarity": similarity,
        })

    results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
