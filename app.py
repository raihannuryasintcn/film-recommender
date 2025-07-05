from flask import Flask, jsonify
import pandas as pd
import joblib
import os
from surprise import Reader, Dataset, SVD

app = Flask(__name__)

# Load data
ratings_df = pd.read_csv("data/ratings.csv")
movies_df = pd.read_csv("data/movies.csv")

# Load model
model = joblib.load("models/svd_model.joblib")

# Fungsi rekomendasi
def recommend_movies_for_user(user_id, ratings_df, movies_df, model, n=10):
    all_movie_ids = movies_df['movieId'].unique()
    watched_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    unseen_movie_ids = [m for m in all_movie_ids if m not in watched_movie_ids]
    
    predictions = [model.predict(user_id, movie_id) for movie_id in unseen_movie_ids]
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    
    top_movie_ids = [pred.iid for pred in top_predictions]
    recommended = movies_df[movies_df['movieId'].isin(top_movie_ids)][['movieId', 'title']]
    
    return recommended.to_dict(orient="records")

# Endpoint rekomendasi
@app.route("/recommend/<int:user_id>", methods=["GET"])
def recommend(user_id):
    try:
        recommendations = recommend_movies_for_user(user_id, ratings_df, movies_df, model)
        return jsonify({"userId": user_id, "recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run
if __name__ == "__main__":
    app.run(debug=True)
