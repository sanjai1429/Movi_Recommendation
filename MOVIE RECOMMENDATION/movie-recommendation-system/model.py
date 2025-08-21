import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Sample moods and movies
movies_by_mood = {
    "Happy": ["Zootopia", "The Intouchables", "La La Land", "Paddington", "Coco"],
    "Sad": ["The Pursuit of Happyness", "Marley & Me", "A Beautiful Mind", "Manchester by the Sea", "Room"],
    "Excited": ["Mad Max: Fury Road", "Inception", "Avengers: Endgame", "The Dark Knight", "Gladiator"],
    "Relaxed": ["Before Sunrise", "Chef", "The Secret Life of Walter Mitty", "Eat Pray Love", "Finding Nemo"],
    "Angry": ["John Wick", "Kill Bill", "Gladiator", "The Revenant", "V for Vendetta"],
    "Romantic": ["The Notebook", "Pride & Prejudice", "Titanic", "Me Before You", "Crazy Rich Asians"]
}

# Create dataset
data = []
for mood, movies in movies_by_mood.items():
    for movie in movies:
        data.append({"Mood": mood, "Movie": movie})

df = pd.DataFrame(data)

# Encode labels
mood_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

df["MoodEncoded"] = mood_encoder.fit_transform(df["Mood"])
df["MovieEncoded"] = movie_encoder.fit_transform(df["Movie"])

# Train model
X = df[["MoodEncoded"]]
y = df["MovieEncoded"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model + encoders
os.makedirs("mood_app", exist_ok=True)
joblib.dump(model, "mood_app/mood_movie_model.pkl")
joblib.dump(mood_encoder, "mood_app/mood_encoder.pkl")
joblib.dump(movie_encoder, "mood_app/movie_encoder.pkl")

print("Model training complete. Files saved in mood_app/")
