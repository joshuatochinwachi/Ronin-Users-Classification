"""
Ronin Trader Classification - Flask API
Author: Jo$h

REST API for making predictions on Ronin trader data.
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np
import traceback
import os

app = Flask(__name__)

# Global variables for model and features
model = None
feature_names = None

def load_model():
    """Load the trained model and feature names at startup."""
    global model, feature_names
    
    model_path = os.getenv('MODEL_PATH', 'models/best_model_random_forest.pkl')
    features_path = os.getenv('FEATURES_PATH', 'models/feature_names.pkl')
    
    print("Loading model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Loading feature names...")
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"✅ Model loaded successfully!")
    print(f"✅ Feature names: {feature_names}")

# Load model when app starts
load_model()

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information."""
    return jsonify({
        'service': 'Ronin Trader Classification API',
        'version': '1.0',
        'author': 'Jo$h',
        'description': 'Predict whether a Ronin blockchain user will remain active',
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Make a single prediction',
            '/predict_batch': 'POST - Make batch predictions'
        },
        'model': 'Random Forest',
        'model_performance': {
            'accuracy': '91.4%',
            'roc_auc': '0.9646'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_loaded': feature_names is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict for a single trader.
    
    Expected JSON input:
    {
        "tx_count_365d": 150,
        "total_volume": 25.5,
        "active_weeks": 20,
        "avg_tx_value": 0.17,
        "tx_per_active_week": 7.5
    }
    
    Returns:
    {
        "prediction": "Good Trader",
        "will_remain_active": true,
        "confidence": 0.95,
        "probability_good_trader": 0.95,
        "probability_bad_trader": 0.05
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required features
        missing_features = set(feature_names) - set(data.keys())
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing': list(missing_features),
                'required_features': feature_names
            }), 400
        
        # Validate feature values
        for feature in feature_names:
            if not isinstance(data[feature], (int, float)):
                return jsonify({
                    'error': f'Feature "{feature}" must be a number',
                    'received_type': type(data[feature]).__name__
                }), 400
            
            if data[feature] < 0:
                return jsonify({
                    'error': f'Feature "{feature}" cannot be negative',
                    'received_value': data[feature]
                }), 400
        
        # Create feature array in correct order
        features = np.array([[data[feat] for feat in feature_names]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Prepare response
        result = {
            'prediction': 'Good Trader' if prediction == 1 else 'Bad Trader',
            'will_remain_active': bool(prediction),
            'confidence': float(max(probability)),
            'probability_good_trader': float(probability[1]),
            'probability_bad_trader': float(probability[0]),
            'input_features': data
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict for multiple traders.
    
    Expected JSON input:
    {
        "traders": [
            {
                "tx_count_365d": 150,
                "total_volume": 25.5,
                "active_weeks": 20,
                "avg_tx_value": 0.17,
                "tx_per_active_week": 7.5
            },
            {
                "tx_count_365d": 10,
                "total_volume": 0.5,
                "active_weeks": 2,
                "avg_tx_value": 0.05,
                "tx_per_active_week": 5.0
            }
        ]
    }
    
    Returns:
    {
        "predictions": [
            {
                "prediction": "Good Trader",
                "confidence": 0.95,
                ...
            },
            ...
        ],
        "summary": {
            "total": 2,
            "good_traders": 1,
            "bad_traders": 1
        }
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'traders' not in data:
            return jsonify({'error': 'No traders data provided'}), 400
        
        traders = data['traders']
        
        if not isinstance(traders, list):
            return jsonify({'error': 'traders must be a list'}), 400
        
        if len(traders) == 0:
            return jsonify({'error': 'traders list is empty'}), 400
        
        # Process each trader
        results = []
        for i, trader in enumerate(traders):
            # Validate features
            missing_features = set(feature_names) - set(trader.keys())
            if missing_features:
                return jsonify({
                    'error': f'Trader at index {i} is missing features',
                    'missing': list(missing_features)
                }), 400
            
            # Create feature array
            features = np.array([[trader[feat] for feat in feature_names]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            results.append({
                'index': i,
                'prediction': 'Good Trader' if prediction == 1 else 'Bad Trader',
                'will_remain_active': bool(prediction),
                'confidence': float(max(probability)),
                'probability_good_trader': float(probability[1]),
                'probability_bad_trader': float(probability[0]),
                'input_features': trader
            })
        
        # Summary statistics
        good_traders = sum(1 for r in results if r['will_remain_active'])
        bad_traders = len(results) - good_traders
        
        return jsonify({
            'predictions': results,
            'summary': {
                'total': len(results),
                'good_traders': good_traders,
                'bad_traders': bad_traders,
                'percentage_good': round(good_traders / len(results) * 100, 2)
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Return the required feature names and descriptions."""
    return jsonify({
        'required_features': feature_names,
        'descriptions': {
            'tx_count_365d': 'Total number of transactions in the past 365 days',
            'total_volume': 'Total transaction volume in USD',
            'active_weeks': 'Number of weeks the user was active',
            'avg_tx_value': 'Average value per transaction in USD',
            'tx_per_active_week': 'Average transactions per active week'
        },
        'example': {
            'tx_count_365d': 150,
            'total_volume': 25.5,
            'active_weeks': 20,
            'avg_tx_value': 0.17,
            'tx_per_active_week': 7.5
        }
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)