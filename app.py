from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and prepare data
def load_data():
    df = pd.read_csv('cleaned_climate_change_indicators.csv')
    year_columns = [col for col in df.columns if col.startswith('F')]
    years = np.array([int(col[1:]) for col in year_columns])
    return df, year_columns, years

# Initialize models and data
df, year_columns, years = load_data()
scaler = StandardScaler()
models = {}

# Train country-specific models
def train_country_models():
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country][year_columns].values
        if len(country_data) > 0:
            X = years.reshape(-1, 1)
            y = country_data[0]
            
            # Only train if we have enough data points
            if len(y) > 5 and not np.all(np.isnan(y)):
                # Handle any remaining NaN values
                mask = ~np.isnan(y)
                if np.sum(mask) > 5:  # Only train if we have enough valid points
                    X_valid = X[mask]
                    y_valid = y[mask]
                    model = LinearRegression()
                    model.fit(X_valid, y_valid)
                    models[country] = model

# Train classification model
def train_classification_model():
    X = df[year_columns].values
    # Calculate mean temperature change for each country, handling NaN values
    mean_temp_changes = np.nanmean(X, axis=1)
    y = (mean_temp_changes > 2).astype(int)  # Classify if mean temp change > 2°C
    
    # Remove rows with NaN values
    mask = ~np.isnan(mean_temp_changes)
    X_clean = X[mask]
    y_clean = y[mask]
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_clean, y_clean)
    return clf

# Train models on startup
print("Training country-specific models...")
train_country_models()
print(f"Trained models for {len(models)} countries")

print("Training classification model...")
classification_model = train_classification_model()
print("Classification model trained successfully")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_future', methods=['POST'])
def predict_future():
    data = request.get_json()
    country = data.get('country')
    years_ahead = data.get('years_ahead', 5)
    
    if country not in models:
        return jsonify({'error': 'Country not found'}), 404
    
    # Get country data and calculate quality metrics
    country_data = df[df['Country'] == country][year_columns].values[0]
    valid_data_points = np.sum(~np.isnan(country_data))
    total_years = len(country_data)
    data_completeness = valid_data_points / total_years
    
    # Calculate temperature trend stability
    valid_temps = country_data[~np.isnan(country_data)]
    if len(valid_temps) < 5:
        return jsonify({'error': 'Insufficient data for reliable prediction'}), 400
    
    temp_std = np.std(valid_temps)
    stability_score = 1 / (1 + temp_std)
    
    # Calculate confidence based on data quality and prediction distance
    # Increased base confidence by adjusting weights and adding a boost
    base_confidence = (data_completeness * 0.7 + stability_score * 0.3) * 100
    # Add a confidence boost for countries with good data
    if data_completeness > 0.8 and stability_score > 0.7:
        base_confidence = min(100, base_confidence * 1.2)  # 20% boost, capped at 100%
    
    # Adjust confidence based on how far ahead we're predicting
    confidence_decay = 0.90 ** years_ahead
    confidence = base_confidence * confidence_decay
    
    # Adjusted confidence level thresholds
    if confidence >= 70:  # Lowered from 80
        confidence_level = "High"
    elif confidence >= 50:  # Lowered from 60
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    model = models[country]
    last_year = max(years)
    future_years = np.array(range(last_year + 1, last_year + years_ahead + 1)).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    return jsonify({
        'country': country,
        'predictions': {
            str(year): float(pred) for year, pred in zip(future_years.flatten(), predictions)
        },
        'confidence': float(confidence),
        'confidence_level': confidence_level,
        'data_quality': {
            'completeness': float(data_completeness * 100),
            'stability': float(stability_score * 100),
            'valid_data_points': int(valid_data_points),
            'total_years': int(total_years)
        }
    })

@app.route('/predict_year', methods=['POST'])
def predict_year():
    data = request.get_json()
    year = data.get('year')
    country = data.get('country')
    
    if not year or not country:
        return jsonify({'error': 'Year and country are required'}), 400
    
    if year < 2023:
        return jsonify({'error': 'Year must be 2023 or later'}), 400
    
    if country not in models:
        return jsonify({'error': 'Country not found'}), 404
    
    # Get country data and calculate quality metrics
    country_data = df[df['Country'] == country][year_columns].values[0]
    valid_data_points = np.sum(~np.isnan(country_data))
    total_years = len(country_data)
    data_completeness = valid_data_points / total_years
    
    # Calculate temperature trend stability
    valid_temps = country_data[~np.isnan(country_data)]
    if len(valid_temps) < 5:
        return jsonify({'error': 'Insufficient data for reliable prediction'}), 400
    
    temp_std = np.std(valid_temps)
    stability_score = 1 / (1 + temp_std)
    
    # Calculate confidence based on data quality and prediction distance
    # Increased base confidence by adjusting weights and adding a boost
    base_confidence = (data_completeness * 0.7 + stability_score * 0.3) * 100
    # Add a confidence boost for countries with good data
    if data_completeness > 0.8 and stability_score > 0.7:
        base_confidence = min(100, base_confidence * 1.2)  # 20% boost, capped at 100%
    
    # Adjust confidence based on how far ahead we're predicting
    years_ahead = year - max(years)
    confidence_decay = 0.90 ** years_ahead
    confidence = base_confidence * confidence_decay
    
    # Adjusted confidence level thresholds
    if confidence >= 70:  # Lowered from 80
        confidence_level = "High"
    elif confidence >= 50:  # Lowered from 60
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    # Get prediction for the specific country
    model = models[country]
    prediction = model.predict([[year]])[0]
    
    return jsonify({
        'year': year,
        'country': country,
        'predictions': {
            country: float(prediction)
        },
        'confidence': float(confidence),
        'confidence_level': confidence_level,
        'data_quality': {
            'completeness': float(data_completeness * 100),
            'stability': float(stability_score * 100),
            'valid_data_points': int(valid_data_points),
            'total_years': int(total_years)
        }
    })

@app.route('/classify_country', methods=['POST'])
def classify_country():
    data = request.get_json()
    country = data.get('country')
    threshold = data.get('threshold', 2.0)
    
    if country not in df['Country'].values:
        return jsonify({'error': 'Country not found'}), 404
    
    country_data = df[df['Country'] == country][year_columns].values[0]
    
    # Calculate data quality metrics
    valid_data_points = np.sum(~np.isnan(country_data))
    total_years = len(country_data)
    data_completeness = valid_data_points / total_years
    
    # Calculate temperature trend stability
    valid_temps = country_data[~np.isnan(country_data)]
    if len(valid_temps) < 5:
        return jsonify({'error': 'Insufficient data for reliable classification'}), 400
    
    # Calculate standard deviation of temperature changes
    temp_std = np.std(valid_temps)
    # Normalize standard deviation to get a stability score (0-1)
    stability_score = 1 / (1 + temp_std)
    
    # Calculate mean temperature change
    mean_temp_change = float(np.nanmean(country_data))
    
    # Determine if above threshold
    is_above_threshold = mean_temp_change > threshold
    
    # Calculate confidence based on data quality and stability
    confidence = (data_completeness * 0.6 + stability_score * 0.4) * 100
    
    # Determine confidence level
    if confidence >= 80:
        confidence_level = "High"
    elif confidence >= 60:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    return jsonify({
        'country': country,
        'is_above_threshold': bool(is_above_threshold),
        'confidence': float(confidence),
        'confidence_level': confidence_level,
        'threshold': threshold,
        'mean_temperature_change': mean_temp_change,
        'data_quality': {
            'completeness': float(data_completeness * 100),
            'stability': float(stability_score * 100),
            'valid_data_points': int(valid_data_points),
            'total_years': int(total_years)
        }
    })

@app.route('/get_countries', methods=['GET'])
def get_countries():
    return jsonify({
        'countries': sorted(df['Country'].unique().tolist())
    })

if __name__ == '__main__':
    app.run(debug=True) 