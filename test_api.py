"""
Test script for Ronin Trader Classification API
Author: Jo$h

Tests all API endpoints with sample data.
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def test_home():
    """Test the home endpoint."""
    print_section("TEST 1: Home Endpoint")
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200, "Home endpoint failed"
    print("✅ Home endpoint test passed!")

def test_health():
    """Test the health check endpoint."""
    print_section("TEST 2: Health Check Endpoint")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200, "Health check failed"
    assert response.json()['status'] == 'healthy', "Service not healthy"
    print("✅ Health check test passed!")

def test_features():
    """Test the features endpoint."""
    print_section("TEST 3: Features Endpoint")
    
    response = requests.get(f"{BASE_URL}/features")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200, "Features endpoint failed"
    print("✅ Features endpoint test passed!")

def test_predict_good_trader():
    """Test prediction for a likely good trader."""
    print_section("TEST 4: Predict - Good Trader Example")
    
    # High activity trader
    data = {
        "tx_count_365d": 500,
        "total_volume": 100.0,
        "active_weeks": 45,
        "avg_tx_value": 0.2,
        "tx_per_active_week": 11.1
    }
    
    print(f"Input data:\n{json.dumps(data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200, "Prediction failed"
    result = response.json()
    print(f"\n📊 Prediction: {result['prediction']}")
    print(f"📊 Confidence: {result['confidence']:.2%}")
    print("✅ Good trader prediction test passed!")

def test_predict_bad_trader():
    """Test prediction for a likely bad trader."""
    print_section("TEST 5: Predict - Bad Trader Example")
    
    # Low activity trader
    data = {
        "tx_count_365d": 5,
        "total_volume": 0.1,
        "active_weeks": 2,
        "avg_tx_value": 0.02,
        "tx_per_active_week": 2.5
    }
    
    print(f"Input data:\n{json.dumps(data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200, "Prediction failed"
    result = response.json()
    print(f"\n📊 Prediction: {result['prediction']}")
    print(f"📊 Confidence: {result['confidence']:.2%}")
    print("✅ Bad trader prediction test passed!")

def test_predict_batch():
    """Test batch prediction."""
    print_section("TEST 6: Batch Prediction")
    
    data = {
        "traders": [
            {
                "tx_count_365d": 500,
                "total_volume": 100.0,
                "active_weeks": 45,
                "avg_tx_value": 0.2,
                "tx_per_active_week": 11.1
            },
            {
                "tx_count_365d": 10,
                "total_volume": 0.5,
                "active_weeks": 2,
                "avg_tx_value": 0.05,
                "tx_per_active_week": 5.0
            },
            {
                "tx_count_365d": 250,
                "total_volume": 50.0,
                "active_weeks": 30,
                "avg_tx_value": 0.2,
                "tx_per_active_week": 8.3
            }
        ]
    }
    
    print(f"Input: {len(data['traders'])} traders")
    
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    result = response.json()
    
    print(f"\nSummary:\n{json.dumps(result['summary'], indent=2)}")
    
    print("\nPredictions:")
    for pred in result['predictions']:
        print(f"  Trader {pred['index']}: {pred['prediction']} "
              f"(confidence: {pred['confidence']:.2%})")
    
    assert response.status_code == 200, "Batch prediction failed"
    print("✅ Batch prediction test passed!")

def test_invalid_input():
    """Test error handling with invalid input."""
    print_section("TEST 7: Error Handling - Invalid Input")
    
    # Missing required feature
    data = {
        "tx_count_365d": 100,
        "total_volume": 10.0
        # Missing: active_weeks, avg_tx_value, tx_per_active_week
    }
    
    print(f"Input data (intentionally incomplete):\n{json.dumps(data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 400, "Should return 400 for invalid input"
    print("✅ Error handling test passed!")

def test_negative_values():
    """Test error handling with negative values."""
    print_section("TEST 8: Error Handling - Negative Values")
    
    data = {
        "tx_count_365d": -10,  # Invalid: negative
        "total_volume": 10.0,
        "active_weeks": 5,
        "avg_tx_value": 2.0,
        "tx_per_active_week": 2.0
    }
    
    print(f"Input data (with negative value):\n{json.dumps(data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 400, "Should return 400 for negative values"
    print("✅ Negative values test passed!")

def run_all_tests():
    """Run all tests."""
    print("="*80)
    print("RONIN TRADER CLASSIFICATION API - TEST SUITE")
    print("="*80)
    
    try:
        test_home()
        test_health()
        test_features()
        test_predict_good_trader()
        test_predict_bad_trader()
        test_predict_batch()
        test_invalid_input()
        test_negative_values()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API.")
        print("Make sure the API is running at http://localhost:5000")
        print("Run: python app.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    run_all_tests()