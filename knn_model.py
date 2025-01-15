import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load your dataset
data = pd.read_csv("spotify_synthetic_data.csv")

# Preprocessing
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features])

# Train KNN model
knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn.fit(data_scaled)

# Save the model
with open("knn_model.pkl", "wb") as file:
    pickle.dump(knn, file)

print("Model saved successfully!")
