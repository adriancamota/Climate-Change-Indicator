from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and prepare data
def load_data():
    df = pd.read_csv('cleaned_climate_change_indicators.csv')
    year_columns = [col for col in df.columns if col.startswith('F')]
    years = np.array([int(col[1:]) for col in year_columns])
    
    # Check if any year columns were found
    if len(year_columns) == 0:
        print("Error: No year columns (starting with 'F') found in the data file.")
        # Depending on desired behavior, you might return None or raise an exception here
        # For now, we'll proceed, but subsequent operations relying on 'years' might fail.
    
    return df, year_columns, years

# Initialize models and data
df, year_columns, years = load_data()

# Add a check here to stop if data loading failed or years is empty
if years is None or len(years) == 0:
    print("Application will not start because historical year data could not be loaded.")
    # You might want to exit or handle this more gracefully in a production app
    # For now, we'll let Flask continue, but API endpoints will likely fail.

scaler = StandardScaler()
models = {}

# Train country-specific models
def train_country_models():
    model_metrics = {}
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
                    
                    # Perform cross-validation
                    model = LinearRegression()
                    cv_scores = cross_val_score(model, X_valid, y_valid, cv=min(5, len(X_valid)), scoring='neg_mean_squared_error')
                    rmse_scores = np.sqrt(-cv_scores)
                    
                    # Train the final model
                    model.fit(X_valid, y_valid)
                    
                    # Calculate additional metrics
                    y_pred = model.predict(X_valid)
                    r2 = r2_score(y_valid, y_pred)
                    mse = mean_squared_error(y_valid, y_pred)
                    
                    models[country] = model
                    model_metrics[country] = {
                        'rmse_cv': float(np.mean(rmse_scores)),
                        'rmse_std': float(np.std(rmse_scores)),
                        'r2_score': float(r2),
                        'mse': float(mse)
                    }
    
    return model_metrics

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
model_metrics = train_country_models()
print(f"Trained models for {len(models)} countries")

print("Training classification model...")
classification_model = train_classification_model()
print("Classification model trained successfully")

# Add regional grouping
REGIONAL_GROUPS = {
    'Europe': ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'Netherlands', 'Belgium', 'Switzerland', 'Austria', 'Sweden', 'Norway', 'Denmark', 'Finland', 'Poland', 'Greece', 'Portugal'],
    'North America': ['United States', 'Canada', 'Mexico'],
    'South America': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela', 'Ecuador'],
    'Asia': ['China', 'Japan', 'India', 'South Korea', 'Indonesia', 'Thailand', 'Vietnam', 'Malaysia', 'Philippines', 'Singapore'],
    'Africa': ['South Africa', 'Egypt', 'Nigeria', 'Kenya', 'Morocco', 'Ethiopia', 'Tanzania'],
    'Oceania': ['Australia', 'New Zealand', 'Fiji', 'Papua New Guinea']
}

def get_region(country):
    for region, countries in REGIONAL_GROUPS.items():
        if country in countries:
            return region
    return 'Other'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_future', methods=['POST'])
def predict_future():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        country = data.get('country')
        years_ahead = data.get('years_ahead', 5)
        
        if not country:
            return jsonify({'error': 'Country is required'}), 400
            
        if country not in models:
            print(f"Country not found in models: {country}")
            return jsonify({'error': 'Country not found'}), 404
        
        # Get country data and calculate quality metrics
        country_data = df[df['Country'] == country][year_columns].values
        if len(country_data) == 0:
            print(f"No data found for country: {country}")
            return jsonify({'error': 'No data found for country'}), 404
            
        country_data = country_data[0]
        valid_data_points = np.sum(~np.isnan(country_data))
        total_years = len(country_data)
        data_completeness = valid_data_points / total_years
        
        # Calculate temperature trend stability
        valid_temps = country_data[~np.isnan(country_data)]
        if len(valid_temps) < 5:
            print(f"Insufficient data points for country {country}: {len(valid_temps)}")
            return jsonify({'error': 'Insufficient data for reliable prediction'}), 400
        
        temp_std = np.std(valid_temps)
        stability_score = 1 / (1 + temp_std)
        
        # Get model metrics
        metrics = model_metrics.get(country, {})
        
        # Calculate confidence based on data quality, prediction distance, and model metrics
        base_confidence = (data_completeness * 0.5 + stability_score * 0.3 + (1 - metrics.get('rmse_cv', 1)) * 0.2) * 100
        
        # Add a confidence boost for countries with good data and model performance
        if data_completeness > 0.8 and stability_score > 0.7 and metrics.get('r2_score', 0) > 0.7:
            base_confidence = min(100, base_confidence * 1.2)
        
        # Adjust confidence based on how far ahead we're predicting
        print(f"Years ahead received in predict_future: {years_ahead}")
        confidence_decay = 0.90 ** years_ahead
        confidence = base_confidence * confidence_decay
        
        # Adjusted confidence level thresholds
        if confidence >= 70:
            confidence_level = "High"
        elif confidence >= 50:
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
            },
            'model_metrics': {
                'rmse': float(metrics.get('rmse_cv', 0)),
                'r2_score': float(metrics.get('r2_score', 0)),
                'prediction_uncertainty': float(metrics.get('rmse_std', 0))
            }
        })
    except Exception as e:
        print(f"Error in predict_future: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/predict_year', methods=['POST'])
def predict_year():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        year = data.get('year')
        country = data.get('country')
        
        if not year or not country:
            return jsonify({'error': 'Year and country are required'}), 400
        
        # Ensure year is an integer
        try:
            year = int(year)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid year provided. Year must be an integer.'}), 400
            
        if year < 2023:
            return jsonify({'error': 'Year must be 2023 or later'}), 400
        
        if country not in models:
            print(f"Country not found in models: {country}")
            return jsonify({'error': 'Country not found'}), 404
        
        # Get country data and calculate quality metrics
        country_data = df[df['Country'] == country][year_columns].values
        if len(country_data) == 0:
            print(f"No data found for country: {country}")
            return jsonify({'error': 'No data found for country'}), 404
            
        country_data = country_data[0]
        valid_data_points = np.sum(~np.isnan(country_data))
        total_years = len(country_data)
        data_completeness = valid_data_points / total_years
        
        # Calculate temperature trend stability
        valid_temps = country_data[~np.isnan(country_data)]
        if len(valid_temps) < 5:
            print(f"Insufficient data points for country {country}: {len(valid_temps)}")
            return jsonify({'error': 'Insufficient data for reliable prediction'}), 400
        
        temp_std = np.std(valid_temps)
        stability_score = 1 / (1 + temp_std)
        
        # Get model metrics
        metrics = model_metrics.get(country, {})
        
        # Calculate confidence based on data quality, prediction distance, and model metrics
        base_confidence = (data_completeness * 0.5 + stability_score * 0.3 + (1 - metrics.get('rmse_cv', 1)) * 0.2) * 100
        
        # Add a confidence boost for countries with good data and model performance
        if data_completeness > 0.8 and stability_score > 0.7 and metrics.get('r2_score', 0) > 0.7:
            base_confidence = min(100, base_confidence * 1.2)
        
        # Adjust confidence based on how far ahead we're predicting
        years_ahead = year - max(years)
        # Ensure years_ahead is not None before exponentiation
        if years_ahead is None:
             return jsonify({'error': 'Could not calculate years ahead for prediction.'}), 500
             
        confidence_decay = 0.90 ** years_ahead
        confidence = base_confidence * confidence_decay
        
        # Adjusted confidence level thresholds
        if confidence >= 70:
            confidence_level = "High"
        elif confidence >= 50:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Get prediction for the specific year
        model = models[country]
        prediction = model.predict([[year]])[0]
        
        return jsonify({
            'year': year,
            'country': country,
            'predictions': {
                str(year): float(prediction)
            },
            'confidence': float(confidence),
            'confidence_level': confidence_level,
            'data_quality': {
                'completeness': float(data_completeness * 100),
                'stability': float(stability_score * 100),
                'valid_data_points': int(valid_data_points),
                'total_years': int(total_years)
            },
            'model_metrics': {
                'rmse': float(metrics.get('rmse_cv', 0)),
                'r2_score': float(metrics.get('r2_score', 0)),
                'prediction_uncertainty': float(metrics.get('rmse_std', 0))
            }
        })
    except Exception as e:
        print(f"Error in predict_year: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

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

@app.route('/model_accuracy', methods=['GET'])
def model_accuracy():
    try:
        # Create a DataFrame with model metrics
        accuracy_data = []
        for country in models.keys():
            try:
                metrics = model_metrics.get(country, {})
                country_data = df[df['Country'] == country][year_columns].values[0]
                valid_data_points = np.sum(~np.isnan(country_data))
                total_years = len(country_data)
                data_completeness = valid_data_points / total_years if total_years > 0 else 0
                
                # Calculate temperature trend stability
                valid_temps = country_data[~np.isnan(country_data)]
                temp_std = np.std(valid_temps) if len(valid_temps) > 0 else 0
                stability_score = 1 / (1 + temp_std) if temp_std > 0 else 0
                
                accuracy_data.append({
                    'Country': country,
                    'Region': get_region(country),
                    'R² Score': float(metrics.get('r2_score', 0) * 100),
                    'RMSE': float(metrics.get('rmse_cv', 0)),
                    'Prediction Uncertainty': float(metrics.get('rmse_std', 0)),
                    'Data Completeness (%)': float(data_completeness * 100),
                    'Temperature Stability (%)': float(stability_score * 100),
                    'Valid Data Points': int(valid_data_points),
                    'Total Years': int(total_years)
                })
            except Exception as e:
                print(f"Error processing country {country}: {str(e)}")
                continue
        
        if not accuracy_data:
            return jsonify({'error': 'No accuracy data available'}), 404
        
        # Create DataFrame and sort by R² Score
        accuracy_df = pd.DataFrame(accuracy_data)
        accuracy_df = accuracy_df.sort_values('R² Score', ascending=False)
        
        # Calculate regional statistics
        regional_stats = accuracy_df.groupby('Region').agg({
            'R² Score': ['mean', 'std', 'count'],
            'RMSE': 'mean',
            'Data Completeness (%)': 'mean',
            'Temperature Stability (%)': 'mean'
        }).round(2)
        
        # Format regional statistics
        regional_stats.columns = ['_'.join(col).strip() for col in regional_stats.columns.values]
        regional_stats = regional_stats.reset_index()
        regional_stats = regional_stats.sort_values('R² Score_mean', ascending=False)
        
        # Format the DataFrames for display
        formatted_df = accuracy_df.round(2)
        formatted_regional = regional_stats.round(2)
        
        # Convert DataFrames to HTML with styling
        html_table = formatted_df.to_html(
            classes='table table-striped table-hover',
            index=False,
            float_format=lambda x: f'{x:.2f}',
            table_id='accuracy-table',
            border=0,
            justify='left'
        )
        
        html_regional = formatted_regional.to_html(
            classes='table table-striped table-hover',
            index=False,
            float_format=lambda x: f'{x:.2f}',
            table_id='regional-table',
            border=0,
            justify='left'
        )
        
        # Add responsive wrappers
        html_table = f'<div class="table-responsive">{html_table}</div>'
        html_regional = f'<div class="table-responsive">{html_regional}</div>'
        
        # Calculate summary statistics
        summary = {
            'total_countries': len(accuracy_df),
            'avg_r2_score': float(accuracy_df['R² Score'].mean()),
            'avg_rmse': float(accuracy_df['RMSE'].mean()),
            'high_accuracy_countries': int(len(accuracy_df[accuracy_df['R² Score'] > 70])),
            'medium_accuracy_countries': int(len(accuracy_df[(accuracy_df['R² Score'] > 50) & (accuracy_df['R² Score'] <= 70)])),
            'low_accuracy_countries': int(len(accuracy_df[accuracy_df['R² Score'] <= 50])),
            'regional_analysis': html_regional
        }
        
        return jsonify({
            'html_table': html_table,
            'summary': summary
        })
    except Exception as e:
        print(f"Error in model_accuracy endpoint: {str(e)}")
        return jsonify({'error': f'Error calculating accuracy metrics: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 