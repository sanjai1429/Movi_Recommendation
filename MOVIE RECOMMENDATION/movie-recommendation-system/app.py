from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load("mood_app/mood_movie_model.pkl")
mood_encoder = joblib.load("mood_app/mood_encoder.pkl")
movie_encoder = joblib.load("mood_app/movie_encoder.pkl")

# Map moods to movie lists (same dataset used to show options)
movies_by_mood = {
    "Happy": ["Zootopia", "The Intouchables", "La La Land", "Paddington", "Coco"],
    "Sad": ["The Pursuit of Happyness", "Marley & Me", "A Beautiful Mind", "Manchester by the Sea", "Room"],
    "Excited": ["Mad Max: Fury Road", "Inception", "Avengers: Endgame", "The Dark Knight", "Gladiator"],
    "Relaxed": ["Before Sunrise", "Chef", "The Secret Life of Walter Mitty", "Eat Pray Love", "Finding Nemo"],
    "Angry": ["John Wick", "Kill Bill", "Gladiator", "The Revenant", "V for Vendetta"],
    "Romantic": ["The Notebook", "Pride & Prejudice", "Titanic", "Me Before You", "Crazy Rich Asians"]
}

@app.route("/", methods=["GET"])
def index():
    moods = list(movies_by_mood.keys())
    return render_template("index.html", moods=moods)

@app.route("/recommend", methods=["POST"])
def recommend():
    mood = request.form.get("mood")
    if mood is None:
        return "Please select a mood", 400
    # encode mood and predict movie index
    mood_enc = mood_encoder.transform([mood])
    pred_idx = model.predict([[int(mood_enc[0])]])[0]
    movie_pred = movie_encoder.inverse_transform([pred_idx])[0]
    # Also pick a random movie from the mood list to diversify results
    import random
    random_choice = random.choice(movies_by_mood[mood])
    return render_template("result.html", mood=mood, movie_pred=movie_pred, random_choice=random_choice)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)