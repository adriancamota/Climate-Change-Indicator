import pandas as pd
import numpy as np
from datetime import datetime

def load_and_validate_data(file_path):
    """
    Load and perform initial validation of the climate data.
    Returns the DataFrame and validation results.
    """
    try:
        df = pd.read_csv(file_path)
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'year_columns': [col for col in df.columns if col.startswith('F')],
            'baseline_period': '1951-1980',  # Document the baseline period
            'data_source': 'FAO',
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }
        return df, validation_results
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def clean_country_names(df):
    """
    Standardize country names and handle special cases.
    """
    country_mapping = {
        'China, P.R.: Mainland': 'China',
        'China, P.R.: Hong Kong': 'Hong Kong',
        'China, P.R.: Macao': 'Macao',
        'Korea, Republic of': 'South Korea',
        'Korea, Democratic People\'s Republic of': 'North Korea',
        'Russian Federation': 'Russia',
        'United States of America': 'United States',
        'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
        'Venezuela (Bolivarian Republic of)': 'Venezuela',
        'Iran (Islamic Republic of)': 'Iran',
        'Syrian Arab Republic': 'Syria',
        'Lao People\'s Democratic Republic': 'Laos',
        'Democratic People\'s Republic of Korea': 'North Korea',
        'Republic of Korea': 'South Korea',
        'United Republic of Tanzania': 'Tanzania',
        'Brunei Darussalam': 'Brunei',
        'Côte d\'Ivoire': 'Ivory Coast',
        'Czech Republic': 'Czechia',
        'Republic of Moldova': 'Moldova',
        'The former Yugoslav Republic of Macedonia': 'North Macedonia',
        'Timor-Leste': 'East Timor',
        'United Arab Emirates': 'UAE',
        'Viet Nam': 'Vietnam'
    }
    
    df['Country'] = df['Country'].replace(country_mapping)
    return df

def handle_missing_values(df):
    """
    Handle missing values with consideration for data quality and bias.
    """
    year_columns = [col for col in df.columns if col.startswith('F')]
    
    # Create a copy of the original data for reference
    df['Original_Data'] = df[year_columns].to_dict('records')
    
    # Forward fill missing values (use previous year's value)
    df[year_columns] = df[year_columns].fillna(method='ffill', axis=1)
    
    # Backward fill any remaining missing values (use next year's value)
    df[year_columns] = df[year_columns].fillna(method='bfill', axis=1)
    
    # For any remaining missing values, fill with the country's mean
    # but only if we have enough data points (more than 50% of years)
    for country in df['Country'].unique():
        country_mask = df['Country'] == country
        country_data = df.loc[country_mask, year_columns]
        
        # Calculate the percentage of non-null values
        data_completeness = country_data.count().mean() / len(year_columns)
        
        if data_completeness > 0.5:  # Only fill if we have more than 50% of the data
            country_mean = country_data.mean().mean()
            df.loc[country_mask, year_columns] = country_data.fillna(country_mean)
        else:
            # Mark countries with insufficient data
            df.loc[country_mask, 'Data_Quality'] = 'Insufficient Data'
    
    return df

def add_data_quality_metrics(df):
    """
    Add data quality metrics to help identify potential biases.
    """
    year_columns = [col for col in df.columns if col.startswith('F')]
    
    # Calculate data completeness for each country
    df['Data_Completeness'] = df[year_columns].count(axis=1) / len(year_columns)
    
    # Calculate the range of temperature changes
    df['Temp_Range'] = df[year_columns].max(axis=1) - df[year_columns].min(axis=1)
    
    # Calculate the standard deviation of temperature changes
    df['Temp_StdDev'] = df[year_columns].std(axis=1)
    
    # Add a flag for countries with extreme values
    df['Has_Extreme_Values'] = (df[year_columns].abs() > 3).any(axis=1)
    
    return df

def save_cleaned_data(df, validation_results):
    """
    Save the cleaned data and metadata.
    """
    # Save the cleaned data
    df.to_csv('cleaned_climate_change_indicators.csv', index=False)
    
    # Save metadata
    metadata = {
        'cleaning_date': datetime.now().strftime('%Y-%m-%d'),
        'original_source': 'FAO',
        'baseline_period': '1951-1980',
        'data_quality_notes': [
            'Temperature changes are relative to 1951-1980 baseline',
            'Some countries may have insufficient data points',
            'Extreme values have been flagged for review',
            'Missing values have been filled using appropriate methods'
        ],
        'validation_results': validation_results
    }
    
    # Save metadata to a separate file
    with open('cleaning_metadata.txt', 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

def main():
    # Load and validate data
    df, validation_results = load_and_validate_data('climate_change_indicators.csv')
    
    if df is not None:
        # Clean the data
        df = clean_country_names(df)
        df = handle_missing_values(df)
        df = add_data_quality_metrics(df)
        
        # Save the results
        save_cleaned_data(df, validation_results)
        
        # Print summary
        print("\nData cleaning completed!")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print("\nData quality metrics:")
        print(f"Countries with insufficient data: {len(df[df['Data_Quality'] == 'Insufficient Data'])}")
        print(f"Countries with extreme values: {len(df[df['Has_Extreme_Values']])}")
        print("\nDetailed results have been saved to 'cleaning_metadata.txt'")

if __name__ == "__main__":
    main() 