import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Keep only required columns
movies = movies[
    [
        "id",
        "title",
        "overview",
        "original_language",
        "genres",
        "vote_average",
        "release_date",
    ]
]

# Remove missing overviews
movies.dropna(subset=["overview"], inplace=True)

# Create TF-IDF vectors
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = tfidf.fit_transform(movies["overview"])

# Create similarity matrix
similarity = cosine_similarity(vectors)

# Save only movies dataframe
pickle.dump(movies, open("movies.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("Movies data prepared successfully!")
print(f"Total movies: {len(movies)}")
print(f"Similarity matrix shape: {similarity.shape}")