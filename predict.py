import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load dataset
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

# Encode userId dan movieId
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings['user'] = user_encoder.fit_transform(ratings['userId'])
ratings['movie'] = movie_encoder.fit_transform(ratings['movieId'])

num_users = ratings['user'].nunique()
num_movies = ratings['movie'].nunique()

# Siapkan data
X = ratings[['user', 'movie']].values
y = ratings['rating'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model Matrix Factorization
class MovieRecommender(tf.keras.Model):
    def __init__(self, n_users, n_movies, embed_dim=50):
        super().__init__()
        self.user_embed = tf.keras.layers.Embedding(n_users, embed_dim)
        self.movie_embed = tf.keras.layers.Embedding(n_movies, embed_dim)

    def call(self, inputs):
        user_vec = self.user_embed(inputs[:, 0])
        movie_vec = self.movie_embed(inputs[:, 1])
        dot = tf.reduce_sum(user_vec * movie_vec, axis=1)
        return dot

model = MovieRecommender(num_users, num_movies)
model.compile(optimizer='adam', loss='mse')

# Training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=512)

# Save model dan encoder
os.makedirs("model", exist_ok=True)
model.save("model/tf_model.h5")
np.save("model/user_encoder.npy", user_encoder.classes_)
np.save("model/movie_encoder.npy", movie_encoder.classes_)
