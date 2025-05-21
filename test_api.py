import requests
import json

BASE_URL = 'http://localhost:5000'

def test_get_countries():
    """Test the get_countries endpoint"""
    response = requests.get(f'{BASE_URL}/get_countries')
    print("\nTesting GET /get_countries:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_predict_future():
    """Test the predict_future endpoint"""
    data = {
        "country": "United States",
        "years_ahead": 5
    }
    response = requests.post(
        f'{BASE_URL}/predict_future',
        json=data
    )
    print("\nTesting POST /predict_future:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_predict_year():
    """Test the predict_year endpoint"""
    data = {
        "year": 2030
    }
    response = requests.post(
        f'{BASE_URL}/predict_year',
        json=data
    )
    print("\nTesting POST /predict_year:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_classify_country():
    """Test the classify_country endpoint"""
    data = {
        "country": "United States",
        "threshold": 2.0
    }
    response = requests.post(
        f'{BASE_URL}/classify_country',
        json=data
    )
    print("\nTesting POST /classify_country:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == '__main__':
    print("Starting API tests...")
    try:
        test_get_countries()
        test_predict_future()
        test_predict_year()
        test_classify_country()
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the server. Make sure the Flask application is running.")
    except Exception as e:
        print(f"\nError: {str(e)}") 