# 🎵 Song Recommendation System

A content-based song recommender built with **Streamlit**. Pick any song from the dataset, and it suggests 10 similar tracks based on audio features (danceability, energy, tempo, valence, etc.) and genre — complete with album art fetched live from iTunes.

## Features

- 🔍 Searchable dropdown to pick a song from the dataset
- 🎯 Content-based filtering using **cosine similarity** on audio features + genre
- 🖼️ Album art fetched dynamically via the free iTunes Search API (no API key needed)
- ⚡ Fast, cached data loading and similarity computation (`@st.cache_data`)

## How It Works

1. The dataset (`dataset_small.csv`) is loaded and normalized using `MinMaxScaler` on features like `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `valence`, `tempo`, `liveness`, and `popularity`.
2. Genre is one-hot encoded (`track_genre`) and combined with the scaled audio features.
3. A cosine similarity matrix is computed across all songs.
4. When a song is selected, the top 10 most similar songs (excluding itself) are returned, along with their artist and genre.
5. Album artwork for each recommendation is fetched from iTunes based on track name + artist.

## Tech Stack

- **Streamlit** – UI/web app framework
- **Pandas / NumPy** – data handling
- **Scikit-learn** – `MinMaxScaler` + `cosine_similarity`
- **Requests** – iTunes API calls for album art

## Project Structure

```
songRecommender/
├── app.py                 # Main Streamlit app
├── dataset_small.csv      # Pre-filtered, balanced dataset (100 songs/genre)
├── requirements.txt       # Python dependencies
└── .devcontainer/         # Codespaces/devcontainer config
```

## Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
git clone https://github.com/Adeela-saleem/songRecommender.git
cd songRecommender
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Dataset

`dataset_small.csv` contains pre-filtered and balanced song data (sampled ~100 songs per genre) with the following columns:

`track_genre`, `artists`, `album_name`, `popularity`, `energy`, `loudness`, `instrumentalness`, `valence`, `tempo`, `track_name`, `danceability`, `acousticness`, `speechiness`, `liveness`

## Notes

- If album art isn't found on iTunes, a placeholder image is shown instead.
- Recommendations are purely content-based (no user history/ratings involved) — similarity is driven entirely by audio characteristics and genre.

## License

This project is for educational/portfolio purposes.
