"""
Ronin Trader Classification - Prediction Script
Author: Jo$h

This script loads the trained Random Forest model and makes predictions
on new trader data.
"""

import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class RoninTraderPredictor:
    """
    A class to load the trained model and make predictions on Ronin traders.
    """
    
    def __init__(self, model_path='models/best_model_random_forest.pkl', 
                 feature_names_path='models/feature_names.pkl'):
        """
        Initialize the predictor by loading the trained model.
        
        Args:
            model_path (str): Path to the saved model file
            feature_names_path (str): Path to the feature names file
        """
        print("Loading model...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print("Loading feature names...")
        with open(feature_names_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print(f"✅ Model loaded successfully!")
        print(f"✅ Expected features: {self.feature_names}")
    
    def validate_input(self, trader_data):
        """
        Validate that the input data has all required features.
        
        Args:
            trader_data (dict): Dictionary containing trader features
            
        Returns:
            bool: True if valid, raises ValueError if not
        """
        missing_features = set(self.feature_names) - set(trader_data.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Check for negative values
        for feature, value in trader_data.items():
            if feature in self.feature_names and value < 0:
                raise ValueError(f"Feature '{feature}' cannot be negative. Got: {value}")
        
        return True
    
    def predict_single(self, trader_data):
        """
        Make a prediction for a single trader.
        
        Args:
            trader_data (dict): Dictionary with keys:
                - tx_count_365d: Number of transactions in 365 days
                - total_volume: Total transaction volume in USD
                - active_weeks: Number of active weeks
                - avg_tx_value: Average transaction value
                - tx_per_active_week: Transactions per active week
        
        Returns:
            dict: Prediction results with probability scores
        """
        # Validate input
        self.validate_input(trader_data)
        
        # Create feature array in correct order
        features = np.array([[trader_data[feat] for feat in self.feature_names]])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        # Map prediction to label
        label = "Good Trader" if prediction == 1 else "Bad Trader"
        
        result = {
            'prediction': label,
            'will_remain_active': bool(prediction),
            'confidence': float(max(probability)),
            'probability_good_trader': float(probability[1]),
            'probability_bad_trader': float(probability[0]),
            'input_features': trader_data
        }
        
        return result
    
    def predict_batch(self, traders_df):
        """
        Make predictions for multiple traders.
        
        Args:
            traders_df (pd.DataFrame): DataFrame with trader features
        
        Returns:
            pd.DataFrame: DataFrame with predictions and probabilities
        """
        # Validate columns
        missing_cols = set(self.feature_names) - set(traders_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Select features in correct order
        X = traders_df[self.feature_names]
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Create results dataframe
        results_df = traders_df.copy()
        results_df['prediction'] = ['Good Trader' if p == 1 else 'Bad Trader' 
                                     for p in predictions]
        results_df['will_remain_active'] = predictions
        results_df['confidence'] = probabilities.max(axis=1)
        results_df['probability_good_trader'] = probabilities[:, 1]
        results_df['probability_bad_trader'] = probabilities[:, 0]
        
        return results_df


def main():
    """
    Example usage of the predictor.
    """
    print("="*80)
    print("RONIN TRADER CLASSIFICATION - PREDICTION SCRIPT")
    print("="*80)
    
    # Initialize predictor
    predictor = RoninTraderPredictor()
    
    # Example 1: Single trader prediction
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Trader Prediction")
    print("="*80)
    
    trader_example = {
        'tx_count_365d': 150,
        'total_volume': 25.5,
        'active_weeks': 20,
        'avg_tx_value': 0.17,
        'tx_per_active_week': 7.5
    }
    
    print("\nInput trader data:")
    for key, value in trader_example.items():
        print(f"  {key}: {value}")
    
    result = predictor.predict_single(trader_example)
    
    print("\n" + "-"*80)
    print("PREDICTION RESULTS:")
    print("-"*80)
    print(f"Prediction: {result['prediction']}")
    print(f"Will remain active: {result['will_remain_active']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probability (Good Trader): {result['probability_good_trader']:.2%}")
    print(f"Probability (Bad Trader): {result['probability_bad_trader']:.2%}")
    
    # Example 2: Batch prediction
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Prediction (Multiple Traders)")
    print("="*80)
    
    traders_batch = pd.DataFrame([
        {
            'wallet': '0xABC123...',
            'tx_count_365d': 500,
            'total_volume': 100.0,
            'active_weeks': 45,
            'avg_tx_value': 0.2,
            'tx_per_active_week': 11.1
        },
        {
            'wallet': '0xDEF456...',
            'tx_count_365d': 10,
            'total_volume': 0.5,
            'active_weeks': 2,
            'avg_tx_value': 0.05,
            'tx_per_active_week': 5.0
        },
        {
            'wallet': '0xGHI789...',
            'tx_count_365d': 250,
            'total_volume': 50.0,
            'active_weeks': 30,
            'avg_tx_value': 0.2,
            'tx_per_active_week': 8.3
        }
    ])
    
    print("\nInput data for 3 traders:")
    print(traders_batch.to_string(index=False))
    
    results = predictor.predict_batch(traders_batch)
    
    print("\n" + "-"*80)
    print("BATCH PREDICTION RESULTS:")
    print("-"*80)
    print(results[['wallet', 'prediction', 'confidence', 
                   'probability_good_trader']].to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ Prediction complete!")
    print("="*80)


if __name__ == "__main__":
    main()