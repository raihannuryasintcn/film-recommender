from flask import Flask, jsonify, render_template, request
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# --- GLOBAL VARIABLES & MODEL LOADING ---
# Load data
ratings_df = pd.read_csv("data/ratings.csv")
movies_df = pd.read_csv("data/movies.csv")

# Load SVD Model
svd_model = joblib.load("models/svd_model.joblib")

# Load TensorFlow Model
tf_model = tf.keras.models.load_model("models/tf_model.keras")

# Initialize and fit encoders for TF model
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
user_encoder.fit(ratings_df['userId'])
movie_encoder.fit(ratings_df['movieId'])
# --- END OF GLOBAL VARIABLES ---


# --- RECOMMENDATION FUNCTIONS ---
def recommend_movies_for_user_svd(user_id, ratings_df, movies_df, model, n=10):
    all_movie_ids = movies_df['movieId'].unique()
    watched_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    unseen_movie_ids = [m for m in all_movie_ids if m not in watched_movie_ids]
    
    predictions = [model.predict(user_id, movie_id) for movie_id in unseen_movie_ids]
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    
    top_movie_ids = [pred.iid for pred in top_predictions]
    recommended_df = movies_df[movies_df['movieId'].isin(top_movie_ids)]
    
    # Add predicted rating to the result
    recommendations = []
    for _, row in recommended_df.iterrows():
        pred_rating = next(pred.est for pred in top_predictions if pred.iid == row['movieId'])
        recommendations.append({
            'movieId': int(row['movieId']),
            'title': row['title'],
            'predicted_rating': round(pred_rating, 2)
        })
    
    # Sort final result by predicted rating
    recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
    
    return recommendations

def recommend_movies_for_user_tf(user_id, model, movies_df, ratings_df, user_encoder, movie_encoder, n=10):
    all_movie_ids = movies_df['movieId'].unique()
    watched_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    unseen_movie_ids = [m for m in all_movie_ids if m not in watched_movie_ids]
    
    known_unseen_movie_ids = [m for m in unseen_movie_ids if m in movie_encoder.classes_]
    
    if not known_unseen_movie_ids:
        return []

    user_idx = user_encoder.transform([user_id])[0]
    unseen_movie_idx = movie_encoder.transform(known_unseen_movie_ids)
    
    user_input_array = np.array([user_idx] * len(unseen_movie_idx))
    movie_input_array = np.array(unseen_movie_idx)
    
    predictions = model.predict([user_input_array, movie_input_array]).flatten()
    
    results_df = pd.DataFrame({
        'movieId': known_unseen_movie_ids,
        'predicted_rating': predictions
    })
    top_n_results = results_df.sort_values(by='predicted_rating', ascending=False).head(n)
    
    recommendations_df = movies_df.merge(top_n_results, on='movieId')
    
    # Format output to be consistent
    recommendations = []
    for _, row in recommendations_df.iterrows():
        recommendations.append({
            'movieId': int(row['movieId']),
            'title': row['title'],
            'predicted_rating': round(float(row['predicted_rating']), 2)
        })

    return recommendations

# --- FLASK ENDPOINTS ---
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/api/users", methods=["GET"])
def get_users():
    users = sorted(ratings_df['userId'].unique().tolist())
    return jsonify({"users": users})

@app.route("/api/user/<int:user_id>/info", methods=["GET"])
def get_user_info(user_id):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    if user_ratings.empty:
        return jsonify({"error": "User not found"}), 404
    
    total_ratings = len(user_ratings)
    avg_rating = user_ratings['rating'].mean()
    top_watched = user_ratings.merge(movies_df, on='movieId').nlargest(10, 'rating')[['title', 'rating']]
    
    return jsonify({
        "userId": user_id,
        "totalRatings": total_ratings,
        "avgRating": round(avg_rating, 2),
        "topWatched": top_watched.to_dict(orient="records")
    })

@app.route("/api/recommend/<int:user_id>", methods=["GET"])
def recommend(user_id):
    model_type = request.args.get('model', 'svd') # Default to SVD

    try:
        if user_id not in ratings_df['userId'].values:
            return jsonify({"error": "User not found"}), 404
        
        if model_type == 'svd':
            recommendations = recommend_movies_for_user_svd(user_id, ratings_df, movies_df, svd_model)
        elif model_type == 'tf':
            recommendations = recommend_movies_for_user_tf(user_id, tf_model, movies_df, ratings_df, user_encoder, movie_encoder)
        else:
            return jsonify({"error": "Invalid model type specified"}), 400

        return jsonify({
            "userId": user_id, 
            "recommendations": recommendations,
            "total": len(recommendations),
            "modelType": model_type
        })
    except Exception as e:
        # Mengembalikan pesan error yang lebih informatif untuk debugging
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# --- MAIN ---
if __name__ == "__main__":
    app.run(debug=True)
