import requests
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from flask import Flask, request, jsonify

app = Flask(__name__)

def fetch_match_data(api_url, headers=None):
    """Fetch match data from the given API endpoint."""
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Unable to fetch data, Status Code: {response.status_code}")
        return None

def process_match_data(data):
    """Process match data into a structured format."""
    matches = []
    for match in data.get('matches', []):
        match_info = {
            'date': match.get('utcDate', ''),
            'home_team': match.get('homeTeam', {}).get('name', ''),
            'away_team': match.get('awayTeam', {}).get('name', ''),
            'score_home': match.get('score', {}).get('fullTime', {}).get('homeTeam', 0),
            'score_away': match.get('score', {}).get('fullTime', {}).get('awayTeam', 0),
            'corners_home': match.get('statistics', {}).get('corners', {}).get('home', 0),
            'corners_away': match.get('statistics', {}).get('corners', {}).get('away', 0),
            'bookings_home': match.get('statistics', {}).get('bookings', {}).get('home', 0),
            'bookings_away': match.get('statistics', {}).get('bookings', {}).get('away', 0),
            'offsides_home': match.get('statistics', {}).get('offsides', {}).get('home', 0),
            'offsides_away': match.get('statistics', {}).get('offsides', {}).get('away', 0)
        }
        matches.append(match_info)
    return pd.DataFrame(matches)

def train_prediction_model(df):
    """Train a machine learning model to predict match results, goals, corners, offsides, and bookings."""
    df = df.dropna()
    df['result'] = np.where(df['score_home'] > df['score_away'], 1, 
                            np.where(df['score_home'] < df['score_away'], 0, 2))
    features = ['corners_home', 'corners_away', 'bookings_home', 'bookings_away', 'offsides_home', 'offsides_away']
    X = df[features]
    
    # Predicting match result
    y_result = df['result']
    X_train, X_test, y_train, y_test = train_test_split(X, y_result, test_size=0.2, random_state=42)
    result_model = RandomForestClassifier(n_estimators=100, random_state=42)
    result_model.fit(X_train, y_train)
    result_predictions = result_model.predict(X_test)
    result_accuracy = accuracy_score(y_test, result_predictions)
    print(f"Match Result Prediction Accuracy: {result_accuracy * 100:.2f}%")
    
    return result_model

def betting_strategy(predictions, odds):
    """Implement AI-driven betting strategy using Kelly Criterion."""
    bankroll = 1000  # Starting bankroll
    kelly_bets = []
    for prob, odd in zip(predictions, odds):
        edge = (prob * odd - 1) / (odd - 1)
        bet = bankroll * edge
        if bet > 0:
            kelly_bets.append(bet)
        else:
            kelly_bets.append(0)
    return kelly_bets

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict match outcomes and betting recommendations."""
    request_data = request.get_json()
    df_input = pd.DataFrame(request_data['matches'])
    predictions = result_model.predict(df_input[['corners_home', 'corners_away', 'bookings_home', 'bookings_away', 'offsides_home', 'offsides_away']])
    odds = request_data.get('odds', [1.5, 2.0, 3.0])
    bet_recommendations = betting_strategy(predictions, odds)
    
    return jsonify({'predictions': predictions.tolist(), 'recommended_bets': bet_recommendations})

if __name__ == '__main__':
    API_URL = "https://api.football-data.org/v2/competitions/PL/matches"
    HEADERS = {"X-Auth-Token": "your_api_key_here"}  # Replace with an actual API key
    match_data = fetch_match_data(API_URL, HEADERS)
    if match_data:
        df_matches = process_match_data(match_data)
        result_model = train_prediction_model(df_matches)
    app.run(host='0.0.0.0', port=5000)
