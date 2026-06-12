import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

movies = pd.read_csv("movies.csv")

# Keep only needed columns
movies = movies[
    [
        'id',
        'title',
        'overview',
        'original_language',
        'genres',
        'vote_average',
        'release_date'
    ]
]

# Remove rows with missing overview
movies.dropna(inplace=True)

cv = TfidfVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['overview']).toarray()

similarity = cosine_similarity(vectors)

pickle.dump(movies, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Model Created Successfully")