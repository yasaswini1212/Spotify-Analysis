import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import os

# Load Dataset
@st.cache_data
def load_data():
    try:
        # Use relative path for deployment
        file_path = os.path.join(os.path.dirname(__file__), "spotify_synthetic_data.csv")
        return pd.read_csv(file_path, encoding="utf-8")
    except FileNotFoundError:
        st.error("Dataset file not found! Please ensure 'spotify_synthetic_data.csv' is uploaded or included in the app's directory.")
        return None
    except Exception as e:
        st.error(f"Error reading the dataset: {e}")
        return None

# Normalize numerical features
@st.cache_data
def preprocess_data(data):
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo']
    scaler = MinMaxScaler()
    return scaler.fit_transform(data[features])

# Save KNN model to pickle
def save_knn_model(knn, filename='knn_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(knn, file)

# Load the saved KNN model from pickle
def load_knn_model(filename='knn_model.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    return None

# Fit KNN model
@st.cache_resource
def fit_knn(data_scaled):
    knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
    knn.fit(data_scaled)
    save_knn_model(knn)  # Save the model after training
    return knn

# Load and preprocess data
data = load_data()
if data is not None:
    data_scaled = preprocess_data(data)

    # Check if model is already saved, if so, load it
    knn = load_knn_model()

    if knn is None:
        knn = fit_knn(data_scaled)  # If no saved model, train a new one

    # Define `max_recommendations` globally before any slider is used
    max_recommendations = min(len(data) - 1, 15)

    # Add Custom CSS for Styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .spotify-logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100px;
        }
        .spotify-heading {
            font-size: 48px;
            font-weight: bold;
            color: green;
            text-align: center;
            margin-top: 10px;
        }
        .header {
            font-size: 36px;
            font-weight: bold;
            color: green;
            text-align: center;
            margin-top: 20px;
        }
        .subheader {
            font-size: 18px;
            color: #666666;
            text-align: center;
            margin-top: -10px;
            margin-bottom: 30px;
        }
        .recommendations {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .select-label {
            font-size: 22px;
            font-weight: bold;
            color: #333333;
        }
        .stButton > button {
            background-color: green;
            color: white;
            font-weight: bold;
        }
        .stButton > button:active {
            background-color: red;
        }
        .recommended-tracks-heading {
            font-size: 28px;
            font-weight: bold;
            color: #333333;
        }
        .recommended-tracks-by-artist-heading {
            font-size: 32px;
            font-weight: bold;
            color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Add Font Awesome CDN to load icons
    st.markdown("""<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">""", unsafe_allow_html=True)

    # Add Spotify logo and heading
    st.markdown('<img src="https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg" alt="Spotify Logo" class="spotify-logo">', unsafe_allow_html=True)

    # Music Recommendation System Header
    st.markdown('<div class="header">Music Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Find similar music based on audio features</div>', unsafe_allow_html=True)

    # Radio button for toggling between sections
    st.markdown("<span class='select-label'>Choose Recommendation Type:</span>", unsafe_allow_html=True)

    # Define plain text options for the radio button
    recommendation_options = ["🎵 Track-Based Recommendations", "🎤 Artist-Based Recommendations"]

    # Use st.radio with plain text
    selected_option = st.radio("", recommendation_options, index=0, key="recommendation_type")

    # Section 1: Track-Based Recommendations
    if selected_option == "🎵 Track-Based Recommendations":
        st.markdown("<span class='select-label'>Select a Track and Genre:</span>", unsafe_allow_html=True)

        # Combine track name and genre for the selectbox
        data['track_with_genre'] = data['track_name'] + ' - ' + data['genre']
        track_list = data['track_with_genre'].tolist()
        selected_track_genre = st.selectbox("", track_list)

        # Extract the track name and genre from the selected option
        selected_track_name = selected_track_genre.split(' - ')[0]
        selected_genre = selected_track_genre.split(' - ')[1]

        # Get the index of the selected track
        track_idx = data[data['track_name'] == selected_track_name].index[0]

        # Slider for number of recommendations
        st.markdown("<span class='select-label'>Select the number of recommendations:</span>", unsafe_allow_html=True)
        num_recommendations = st.slider("", 1, max_recommendations, 5)

        # Recommend Similar Tracks here
        if st.button("Get Track-Based Recommendations"):
            st.markdown(
                f"🎧Selected Track: **'{selected_track_name}'** (Genre: {selected_genre})",
                unsafe_allow_html=True
            )

            distances, indices = knn.kneighbors([data_scaled[track_idx]], n_neighbors=num_recommendations + 1)

            st.markdown('<div class="recommendations">', unsafe_allow_html=True)
            st.markdown("<i class='fa-solid fa-music fa-2x'></i> <span class='recommended-tracks-heading'>Recommended Tracks:</span>", unsafe_allow_html=True)
            for i in indices[0][1:]:  # Skip the first one as it's the selected track itself
                st.markdown(
                    f"<i class='fa-solid fa-music'></i> **{data.iloc[i]['track_name']}** ({data.iloc[i]['genre']})",
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Artist-Based Recommendation
    elif selected_option == "🎤 Artist-Based Recommendations":
        st.markdown("<span class='select-label'>Select an Artist:</span>", unsafe_allow_html=True)

        # User Input for Artist
        artist_list = data['artist'].unique().tolist()
        selected_artist = st.selectbox("", artist_list)

        # Slider for number of recommendations
        st.markdown("<span class='select-label'>Select the number of recommendations:</span>", unsafe_allow_html=True)
        artist_recommendations = st.slider("", 1, max_recommendations, 5)

        # Recommend Tracks by Artist
        if st.button("Get Artist-Based Recommendations"):
            st.markdown(f"🎤 Selected Artist: **'{selected_artist}'**", unsafe_allow_html=True)

            artist_tracks = data[data['artist'] == selected_artist].head(artist_recommendations)
            if not artist_tracks.empty:
                st.markdown('<div class="recommendations">', unsafe_allow_html=True)
                st.markdown(f"<i class='fa-solid fa-music fa-2x'></i> <span class='recommended-tracks-by-artist-heading'>Recommended Tracks by {selected_artist}:</span>", unsafe_allow_html=True)
                for _, row in artist_tracks.iterrows():
                    st.markdown(
                        f"<i class='fa-solid fa-music'></i> **{row['track_name']}** by ({row['artist']})",
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"No tracks found for the selected artist: **'{selected_artist}'**.", unsafe_allow_html=True)

else:
    st.error("Application cannot proceed without the dataset.")
