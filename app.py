from flask import Flask, render_template, request
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# ---------------------------
# PATHS
# ---------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------
# HOLLYWOOD DATA
# ---------------------------

movies = pd.read_csv(
    os.path.join(BASE_DIR, "movies.csv")
)

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

movies.dropna(subset=["overview"], inplace=True)

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = tfidf.fit_transform(movies["overview"])
similarity = cosine_similarity(vectors)

# ---------------------------
# INDIAN DATA
# ---------------------------

indian_movies_path = os.path.join(
    BASE_DIR,
    "indian movies.csv"
)

indian_movies = pd.read_csv(
    os.path.join(BASE_DIR, "indian movies.csv")
)

indian_movies = indian_movies[
    ["Movie Name", "Genre", "Language", "Rating(10)"]
]

indian_movies = indian_movies[indian_movies["Genre"] != "-"]

indian_movies.dropna(inplace=True)

indian_movies["tags"] = (
    indian_movies["Genre"].astype(str)
    + " "
    + indian_movies["Language"].astype(str)
)

cv = CountVectorizer(stop_words="english")

vectors_indian = cv.fit_transform(
    indian_movies["tags"]
)

indian_similarity = cosine_similarity(vectors_indian)

# ---------------------------
# HOLLYWOOD RECOMMENDER
# ---------------------------

def recommend_hollywood(movie):

    matches = movies[
        movies["title"].str.contains(
            movie,
            case=False,
            na=False
        )
    ]

    if matches.empty:

        suggestions = movies[
            movies["title"].str.contains(
                movie.split()[0],
                case=False,
                na=False
            )
        ]["title"].head(5).tolist()

        if suggestions:
            return ["Did you mean:"] + suggestions

        return ["Movie not found"]

    movie_index = matches.index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movie_list:
        recommendations.append(
            movies.iloc[i[0]]["title"]
        )

    return recommendations

# ---------------------------
# INDIAN RECOMMENDER
# ---------------------------

def recommend_indian(movie, language):

    filtered = indian_movies[
        indian_movies["Language"].str.lower() == language.lower()
    ]

    matches = filtered[
        filtered["Movie Name"].str.contains(
            movie,
            case=False,
            na=False
        )
    ]

    if matches.empty:
        return ["Movie not found"]

    selected_movie = matches.iloc[0]

    genre = selected_movie["Genre"]

    recommendations = filtered[
        filtered["Genre"] == genre
    ]

    recommendations = recommendations.sort_values(
        by="Rating(10)",
        ascending=False
    )

    recommendations = recommendations[
        recommendations["Movie Name"]
        != selected_movie["Movie Name"]
    ]

    return recommendations["Movie Name"].head(5).tolist()

# ---------------------------
# ROUTE
# ---------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []
    movie = ""

    if request.method == "POST":

        category = request.form["category"]
        movie = request.form["movie"]

        if category == "Hollywood":
            recommendations = recommend_hollywood(movie)

        else:
            recommendations = recommend_indian(
                movie,
                category
            )

    return render_template(
    "index.html",
    recommendations=recommendations,
    selected_movie=movie if request.method == "POST" else ""
)

# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    app.run(debug=True)