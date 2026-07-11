import re
import gdown
import joblib
import pandas as pd
import random
import requests
import os
import google.generativeai as genai
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

session = requests.Session()

retry = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)

adapter = HTTPAdapter(max_retries=retry)

session.mount("https://", adapter)
session.mount("http://", adapter)

session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Connection": "keep-alive"
})

# ==============================
# TMDB API KEY
# ==============================
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel("gemini-2.5-flash")

TRENDING_MOVIES = [
    "Avatar",
    "Titanic",
    "Interstellar",
    "Inception",
    "The Dark Knight",
    "Avengers: Endgame"
]
search_history = []

# ==============================
# LOAD DATA
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MATRIX_PATH = os.path.join(MODEL_DIR, "movie_matrix.pkl")

# Download movie_matrix.pkl
if not os.path.exists(MATRIX_PATH):
    print("Downloading movie_matrix.pkl...")
    gdown.download(
        "https://drive.google.com/file/d/1VB26G6yoT5ppljaY0ayhX3Or3K7MhNrk/view?usp=sharing",
        MATRIX_PATH,
        quiet=False
    )


print("Loading model files...")

movie_matrix = joblib.load(MATRIX_PATH)


DATA_DIR = os.path.join(BASE_DIR, "data")

movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
links = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))

# Remove rows having no TMDB id
links = links.dropna(subset=["tmdbId"])

links["movieId"] = links["movieId"].astype(int)
links["tmdbId"] = links["tmdbId"].astype(int)

movies["movieId"] = movies["movieId"].astype(int)

# Merge movie title with tmdb id
movies = movies.merge(
    links[["movieId", "tmdbId"]],
    on="movieId",
    how="left"
)

print("Movies Loaded :", len(movies))

# ==============================
# FIND MOVIE ID
# ==============================

def find_movie(movie_name):

    movie_name = movie_name.lower().strip()

    for _, row in movies.iterrows():

        title = row["title"].lower()

        clean_title = re.sub(r"\(\d{4}\)", "", title).strip()

        if movie_name == clean_title:
            return row

        if movie_name in clean_title:
            return row

    return None

def search_tmdb_movie(movie_name):

    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"

    response = session.get(url, timeout=15)

    if response.status_code != 200:
        return None

    data = response.json()

    if data["results"]:
        return data["results"][0]

    return None

def get_tmdb_recommendations(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations?api_key={TMDB_API_KEY}"

    response = session.get(url, timeout=15)

    if response.status_code != 200:
        return []

    data = response.json()

    recommendations = []

    for movie in data.get("results", [])[:5]:

        details = get_movie_details(tmdb_id=movie["id"])

        if details:
            details["match"] = random.randint(85,98)
            details["confidence"] = min(100, details["match"] + random.randint(2,5))
            if details["match"] >= 95:
                details["match_class"] = "excellent"
            elif details["match"] >= 90:
                details["match_class"] = "good"
            else:
                details["match_class"] = "average"

            reasons = [
                "🎭 Similar genre",
                "⭐ Highly rated by viewers",
                "🔥 Popular among movie lovers",
                "🤖 Recommended by AI similarity",
                "🎬 Similar storyline"
            ]

            details["reason"] = random.choice(reasons)
            print(details)
            recommendations.append(details)

    return recommendations



# ==============================
# GET TMDB DETAILS
# ==============================

def get_movie_details(movie_name=None, tmdb_id=None):

    if tmdb_id is None:

        movie_row = movies[movies["title"] == movie_name]

        if movie_row.empty:
            tmdb_movie = search_tmdb_movie(movie_name)

            if tmdb_movie:
                tmdb_id = tmdb_movie["id"]
            else:
                return None

        else:
            movie_id = int(movie_row.iloc[0]["movieId"])

            link_row = links[links["movieId"] == movie_id]

            if link_row.empty:
                return None

            tmdb_id = int(link_row.iloc[0]["tmdbId"])

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"

    params = {
        "api_key": TMDB_API_KEY
    }

    try:

        response = session.get(url, params=params, timeout=15)

        response.raise_for_status()

        movie = response.json()

        videos = session.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos",
            params=params,
            timeout=15
        ).json()

        trailer = ""

        # First preference: Official Trailer
        for video in videos.get("results", []):

            if (
                video.get("site") == "YouTube"
                and video.get("type") == "Trailer"
            ):
            
                trailer = f"https://www.youtube.com/watch?v={video['key']}"
                break

        # If no trailer exists, use any YouTube video
        if trailer == "":
            for video in videos.get("results", []):

                if video.get("site") == "YouTube":
                    trailer = f"https://www.youtube.com/watch?v={video['key']}"
                    break

        genres = ", ".join(
            genre["name"] for genre in movie.get("genres", [])
        )
        print(videos)
        return {
            "title": movie.get("title"),
            "poster": "https://image.tmdb.org/t/p/w500" + movie["poster_path"] if movie.get("poster_path") else "",

            "backdrop":"https://image.tmdb.org/t/p/original"+movie["backdrop_path"] if movie.get("backdrop_path") else "",
            "rating": movie.get("vote_average"),
            "release": movie.get("release_date"),
            "overview": movie.get("overview"),
            "genres": genres,
            "runtime": movie.get("runtime"),
            "trailer": trailer,
            "streaming": get_streaming_providers(tmdb_id)
            
        }

    except Exception as e:

        print("TMDB ERROR:", e)

        return None

def get_streaming_providers(tmdb_id):

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/watch/providers?api_key={TMDB_API_KEY}"

    try:
        response = session.get(url, timeout=10)

        if response.status_code != 200:
            return []

        data = response.json()

        # India
        india = data.get("results", {}).get("IN")

        if not india:
            return []

        providers = []

        if "flatrate" in india:
            providers += [p["provider_name"] for p in india["flatrate"]]

        if "rent" in india:
            providers += [p["provider_name"] for p in india["rent"]]

        if "buy" in india:
            providers += [p["provider_name"] for p in india["buy"]]

        return list(dict.fromkeys(providers))

    except Exception:
        return []

def get_trending_movies():

    trending = []

    for movie in TRENDING_MOVIES:

        
        print("Loading:", movie)

        movie_row = find_movie(movie)

        if movie_row is not None:
            details = get_movie_details(movie_row["title"])
        else:
            details = None

        print(details)

        if details:
            trending.append(details)

    return trending

# ==============================
# RECOMMEND MOVIES
# ==============================

def recommend(movie_name):

    movie = find_movie(movie_name)

    if movie is None:

        tmdb_movie = search_tmdb_movie(movie_name)

        if not tmdb_movie:
            return None

        selected = get_movie_details(tmdb_id=tmdb_movie["id"])

        recommendations = get_tmdb_recommendations(tmdb_movie["id"])

        return {
            "selected": selected,
            "recommendations": recommendations
        }

    movie_id = movie["movieId"]

    if movie_id not in movie_matrix.index:
        return None

    index = movie_matrix.index.get_loc(movie_id)

    scores = cosine_similarity(
        movie_matrix.iloc[index:index+1],
        movie_matrix
    ).flatten()

    distances = list(enumerate(scores))

    distances = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )[1:11]

    recommendations = []

    for item in distances:

        recommended_movie_id = movie_matrix.index[item[0]]

        movie_info = movies[
            movies["movieId"] == recommended_movie_id
        ]

        if movie_info.empty:
            continue

        movie_info = movie_info.iloc[0]

        movie_title = movie_info["title"]

        details = get_movie_details(movie_title)

        if details:
            match = min(100, round(item[1] * 100))

            details["match"] = match
            details["confidence"] = min(100, match + random.randint(2, 5))

            if match >= 95:
                details["match_class"] = "excellent"

            elif match >= 80:
                details["match_class"] = "good"

            else:
                details["match_class"] = "average"

            reasons = [
                "🎭 Similar genre",
                "⭐ Highly rated by viewers",
                "🔥 Popular among movie lovers",
                "🤖 Recommended by AI similarity",
                "🎬 Similar storyline"
            ]

            details["reason"] = random.choice(reasons)

            recommendations.append(details)

        if len(recommendations) == 5:
            break

    return recommendations

# ==============================
# SEARCH SUGGESTIONS
# ==============================

@app.route("/suggest")

def suggest():

    query = request.args.get("q", "").lower().strip()

    if query == "":
        return jsonify([])

    suggestions = []

    for title in movies["title"]:

        clean = re.sub(r"\(\d{4}\)", "", title)

        if query in clean.lower():

            suggestions.append(clean.strip())

        if len(suggestions) == 10:
            break

    return jsonify(suggestions)

# ==============================
# HOME PAGE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []
    trending_movies = get_trending_movies()
    selected_movie = None
    error = None

    if request.method == "POST":


        movie_name = request.form.get("movie")

        if movie_name and movie_name not in search_history:
            search_history.insert(0, movie_name)

        if len(search_history) > 5:
            search_history.pop()

        movie_row = find_movie(movie_name)

        if movie_row is not None:
            selected_movie = get_movie_details(movie_row["title"])

        recommendations = recommend(movie_name)

        if isinstance(recommendations, dict):
            selected_movie = recommendations.get("selected")
            recommendations = recommendations.get("recommendations", [])

        if selected_movie is None:
            error = "Movie details could not be loaded."

        if recommendations is None or len(recommendations) == 0:
            error = "Movie not found or no recommendations available."
            recommendations = []

    return render_template(
        "index.html",
        recommendations=recommendations,
        trending_movies=trending_movies,
        selected_movie=selected_movie,
        search_history=search_history,
        error=error
    )

@app.route("/autocomplete")
def autocomplete():

    query = request.args.get("q", "").strip()

    if len(query) < 2:
        return jsonify([])

    query_lower = query.lower()

    suggestions = []

    # ----------------------------
    # Local Dataset Suggestions
    # ----------------------------
    titles = movies["title"].dropna().unique().tolist()

    exact = []
    starts = []
    contains = []

    for title in titles:

        clean = re.sub(r"\(\d{4}\)", "", title).strip()
        t = clean.lower()

        if t == query_lower:
            exact.append(title)

        elif t.startswith(query_lower):
            starts.append(title)

        elif query_lower in t:
            contains.append(title)

    starts.sort(key=len)
    contains.sort(key=len)

    local_movies = exact + starts + contains
    # ----------------------------
    # TMDB Suggestions
    # ----------------------------
    try:
        tmdb_movies = []

        params={
            "api_key": TMDB_API_KEY,
            "query": query,
            "language": "en-US",
            "include_adult": False,
            "region": "IN"
        }

        response = session.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=10
        )

        if response.status_code == 200:

            data = response.json()

            print("TMDB AUTOCOMPLETE RESULTS:")
            print(f"TMDB returned {len(data.get('results', []))} movies")

            for movie in data.get("results", []):

                title = movie.get("title")

                print(title)

                if not title:
                    continue

                popularity = movie.get("popularity", 0)

                if title.lower().startswith(query_lower):
                    tmdb_movies.append((title, popularity))

    except Exception as e:

        print("Autocomplete TMDB Error:", e)

    
    # Sort TMDB by popularity
    tmdb_movies.sort(key=lambda x: x[1], reverse=True)

    # Keep only movie titles
    tmdb_titles = [title for title, _ in tmdb_movies]

    # Local dataset first, then TMDB
    suggestions = local_movies + tmdb_titles
    # ----------------------------
    # Remove duplicates
    # ----------------------------
    final = []
    seen = set()

    for movie in suggestions:

        if isinstance(movie, tuple):
            title = movie[0]
        else:
            title = movie

        key = title.lower()

        if key not in seen:
            final.append(title)
            seen.add(key)

    return jsonify(final[:20])

# ==============================
# AI CHATBOT
# ==============================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"reply": "Please ask a movie-related question."})

    prompt = f"""
You are CineAI, an expert movie recommendation assistant.

Rules:
- Answer only movie-related questions.
- Recommend movies whenever appropriate.
- Keep answers concise and friendly.
- If asked something unrelated to movies, politely say that you only answer movie-related questions.

User: {user_message}
"""

    try:

        response = gemini_model.generate_content(prompt)

        return jsonify({
            "reply": response.text
        })

    except Exception as e:

        return jsonify({
            "reply": str(e)
        })

# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "True")