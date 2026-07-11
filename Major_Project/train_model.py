import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

print("Loading datasets...")

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

print("Movies:", movies.shape)
print("Ratings:", ratings.shape)

# Merge ratings with movie information
data = ratings.merge(movies, on="movieId")

# Create matrix using movieId (NOT title)
movie_matrix = data.pivot_table(
    index="movieId",
    columns="userId",
    values="rating"
).fillna(0)

print("Movie Matrix:", movie_matrix.shape)

# Cosine Similarity

os.makedirs("model", exist_ok=True)

joblib.dump(movie_matrix, "model/movie_matrix.pkl")


print("\nModel trained successfully.")
print("Files saved inside model folder.")