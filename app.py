import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Song Recommender", page_icon="🎵", layout="wide")


@st.cache_data
def load_and_process():
    # Pre-filtered, balanced dataset (already cleaned + sampled 100 songs/genre)
    songs = pd.read_csv('dataset_small.csv')

    feature_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                     'acousticness', 'instrumentalness', 'valence',
                     'tempo', 'liveness', 'popularity']
    songs[feature_cols] = MinMaxScaler().fit_transform(songs[feature_cols])

    songs_encoded = pd.get_dummies(songs, columns=['track_genre'])
    genre_cols = [col for col in songs_encoded.columns if col.startswith('track_genre_')]
    all_features = feature_cols + genre_cols

    similarity = cosine_similarity(songs_encoded[all_features])

    return songs_encoded, similarity, genre_cols


def recommend(song_name, songs_encoded, similarity, genre_cols):
    if song_name not in songs_encoded['track_name'].values:
        return []
    idx = songs_encoded[songs_encoded['track_name'] == song_name].index[0]
    distances = similarity[idx]
    songs_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
    results = []
    for i in songs_list:
        row = songs_encoded.iloc[i[0]]
        genre = [col.replace('track_genre_', '') for col in genre_cols if row[col] == 1]
        results.append({
            'track_name': row['track_name'],
            'artists': row['artists'],
            'genre': genre[0] if genre else 'unknown'
        })
    return results


@st.cache_data(show_spinner=False)
def get_album_art(track_name, artist_name):
    """Fetch album art using the free iTunes Search API (no API key needed)."""
    try:
        query = f"{track_name} {artist_name}"
        url = "https://itunes.apple.com/search"
        params = {"term": query, "media": "music", "limit": 1}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        if data.get("resultCount", 0) > 0:
            art_url = data["results"][0]["artworkUrl100"].replace("100x100", "300x300")
            return art_url
    except Exception:
        pass
    return "https://via.placeholder.com/300x300.png?text=No+Image"


# ---- UI ----
st.title("🎵 Song Recommendation System")
st.write("Pick a song you like, and we'll recommend similar tracks based on audio features and genre.")

with st.spinner("Loading and processing data..."):
    songs_encoded, similarity, genre_cols = load_and_process()

song_list = sorted(songs_encoded['track_name'].unique())

st.markdown("### Select a song")
selected_song = st.selectbox(
    label="Search and select a song:",
    options=song_list,
    index=None,
    placeholder="Type to search a song...",
    label_visibility="collapsed"
)

if selected_song:
    if st.button("Get Recommendations 🎧", type="primary"):
        results = recommend(selected_song, songs_encoded, similarity, genre_cols)

        if results:
            st.markdown(f"### Songs similar to *{selected_song}*")
            cols = st.columns(5)
            for i, r in enumerate(results):
                with cols[i % 5]:
                    art_url = get_album_art(r['track_name'], r['artists'])
                    st.image(art_url, use_container_width=True)
                    st.markdown(f"**{r['track_name']}**")
                    st.caption(f"{r['artists']}")
                    st.caption(f"Genre: {r['genre']}")
        else:
            st.error("Song not found!")
else:
    st.info("Choose a song from the dropdown above to get started.")
