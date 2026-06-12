# 🎬 Multi-Language Movie Recommendation System

A Machine Learning-powered Movie Recommendation System that suggests similar movies based on content features such as genres, keywords, cast, and crew information.

The application supports recommendations across multiple film industries, including Hollywood and major Indian cinema industries such as Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, Marathi, and Punjabi movies.

## 🚀 Features

* Content-Based Movie Recommendation Engine
* Hollywood Movie Recommendations using Cosine Similarity
* Multi-Language Indian Movie Recommendations
* Fast Recommendation Generation
* User-Friendly Netflix-Inspired Interface
* Search Movies and Discover Similar Titles
* Machine Learning-Based Similarity Matching

## 🛠️ Tech Stack

### Backend

* Python
* Flask

### Machine Learning & Data Processing

* Pandas
* NumPy
* Scikit-Learn

### Frontend

* HTML5
* CSS3
* Bootstrap 5

## 📂 Project Structure

```text
movie-recommendation-system/
│
├── app.py
├── model/
│   └── recommendation.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── movies.csv
├── credits.csv
├── indian movies.csv
│
├── README.md
└── requirements.txt
```

## ⚙️ Installation

1. Clone the repository

```bash
git clone https://github.com/Arpitasri007/movie-recommendation-system.git
```

2. Navigate to the project directory

```bash
cd movie-recommendation-system
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the application

```bash
python app.py
```

5. Open your browser and visit:

```text
http://127.0.0.1:5000
```

## 🧠 How It Works

The recommendation engine uses a Content-Based Filtering approach:

1. Movie metadata is processed and combined.
2. Text features are transformed into numerical vectors.
3. Cosine Similarity is computed between movies.
4. The system recommends movies that are most similar to the selected title.

## 📈 Future Enhancements

* TMDB API Integration for Movie Posters
* Autocomplete Search Suggestions
* User Authentication System
* Watchlist and Favorites
* Recommendation History
* Hybrid Recommendation Engine
* Cloud Deployment

## 👩‍💻 Author

**Arpita Srivastava**
B.Tech CSE (AI & ML)

GitHub: https://github.com/Arpitasri007
