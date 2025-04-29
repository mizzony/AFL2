# afl_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Page Title
st.title("🏉 AFL Match Winner Predictor")

# Load and Prepare Data
@st.cache_data
def load_data():
    data = pd.read_csv('https://raw.githubusercontent.com/mizzony/AFL/refs/heads/main/afl_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data['Attendance'] = data['Attendance'].str.replace(',', '').astype(float)
    data['Rainfall'] = data['Rainfall'].fillna(data['Rainfall'].median())
    data = data[(data['HomeTeamScore'] >= 0) & (data['AwayTeamScore'] >= 0)]
    
    six_months_ago = data['Date'].max() - pd.Timedelta(days=180)
    data_last_6_months = data[data['Date'] >= six_months_ago]
    
    home_team_avg_points = data_last_6_months.groupby('HomeTeam')['HomeTeamScore'].mean()
    away_team_avg_points = data_last_6_months.groupby('AwayTeam')['AwayTeamScore'].mean()

    data['HomeTeam_PastAvgPoints'] = data['HomeTeam'].map(home_team_avg_points).fillna(0)
    data['AwayTeam_PastAvgPoints'] = data['AwayTeam'].map(away_team_avg_points).fillna(0)

    return data, home_team_avg_points, away_team_avg_points

# Load data
data, home_team_avg_points, away_team_avg_points = load_data()

# Prepare Dataset
X = data[['HomeTeam', 'Year', 'Rainfall', 'Venue', 'HomeTeam_PastAvgPoints', 'AwayTeam', 'AwayTeam_PastAvgPoints']]
y = data['Win']

# Split and Encode
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoders = {}
for col in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# Train Model (Use simple fast parameters for now)
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', seed=42, n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)

# App UI
st.sidebar.header("Choose Match Details")

home_team = st.sidebar.selectbox("Select Home Team", sorted(data['HomeTeam'].unique()))
away_team = st.sidebar.selectbox("Select Away Team", sorted(data['AwayTeam'].unique()))

venue_options = data['Venue'].unique()
venue = st.sidebar.selectbox("Select Venue", sorted(venue_options))

rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 50.0, 0.0)

year = st.sidebar.number_input("Year", min_value=2020, max_value=2025, value=2024)

# Prediction Function
def predict_match(home_team, away_team, venue, rainfall, year):
    if home_team not in home_team_avg_points.index or away_team not in away_team_avg_points.index:
        return "Invalid team selection."

    input_data = pd.DataFrame({
        'HomeTeam': [label_encoders['HomeTeam'].transform([home_team])[0]],
        'Year': [year],
        'Rainfall': [rainfall],
        'Venue': [label_encoders['Venue'].transform([venue])[0]],
        'HomeTeam_PastAvgPoints': [home_team_avg_points.get(home_team, 0)],
        'AwayTeam': [label_encoders['AwayTeam'].transform([away_team])[0]],
        'AwayTeam_PastAvgPoints': [away_team_avg_points.get(away_team, 0)],
    })

    pred = model.predict(input_data)
    return "🏡 Home Team Wins!" if pred[0] == 1 else "🚶‍♂️ Away Team Wins!"

# Predict Button
if st.sidebar.button("Predict Match Outcome"):
    result = predict_match(home_team, away_team, venue, rainfall, year)
    st.success(result)

