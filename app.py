from flask import Flask, jsonify, render_template
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
    
    # Tambahkan predicted rating ke hasil
    recommendations = []
    for _, row in recommended.iterrows():
        pred_rating = next(pred.est for pred in top_predictions if pred.iid == row['movieId'])
        recommendations.append({
            'movieId': int(row['movieId']),
            'title': row['title'],
            'predicted_rating': round(pred_rating, 2)
        })
    
    return recommendations

# Endpoint untuk UI
@app.route("/")
def index():
    return render_template('index.html')

# Endpoint untuk mendapatkan daftar user
@app.route("/api/users", methods=["GET"])
def get_users():
    users = sorted(ratings_df['userId'].unique().tolist())
    return jsonify({"users": users})

# Endpoint untuk mendapatkan info user
@app.route("/api/user/<int:user_id>/info", methods=["GET"])
def get_user_info(user_id):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    if user_ratings.empty:
        return jsonify({"error": "User not found"}), 404
    
    # Statistik user
    total_ratings = len(user_ratings)
    avg_rating = user_ratings['rating'].mean()
    
    # Film yang sudah ditonton (top 10 berdasarkan rating)
    top_watched = user_ratings.merge(movies_df, on='movieId').nlargest(10, 'rating')[['title', 'rating']]
    
    return jsonify({
        "userId": user_id,
        "totalRatings": total_ratings,
        "avgRating": round(avg_rating, 2),
        "topWatched": top_watched.to_dict(orient="records")
    })

# Endpoint rekomendasi
@app.route("/api/recommend/<int:user_id>", methods=["GET"])
def recommend(user_id):
    try:
        # Cek apakah user ada
        if user_id not in ratings_df['userId'].values:
            return jsonify({"error": "User not found"}), 404
            
        recommendations = recommend_movies_for_user(user_id, ratings_df, movies_df, model)
        return jsonify({
            "userId": user_id, 
            "recommendations": recommendations,
            "total": len(recommendations)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run
if __name__ == "__main__":
    app.run(debug=True)