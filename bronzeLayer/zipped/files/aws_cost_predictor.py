"""
AWS Cost Prediction System
---------------------------
Predicts AWS service costs based on historical data using Gradient Boosting model.

Features:
1. Predict cost for specific region, service, and month
2. Predict aggregated cost across all regions for a service and month
"""

import pickle
import pandas as pd
import numpy as np
from typing import Union, Dict

class AWSCostPredictor:
    def __init__(self, model_path: str):
        """Load the trained model and encoders"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.region_encoder = model_data['region_encoder']
        self.service_encoder = model_data['service_encoder']
        self.month_mapping = model_data['month_mapping']
        self.regions = model_data['regions']
        self.services = model_data['services']
    
    def predict_single(self, region: str, service: str, month: str) -> Dict:
        """
        Predict cost for a specific region, service, and month.
        
        Args:
            region: Region name (e.g., 'US Region', 'APAC Region', 'EU Region')
            service: Service name (e.g., 'EC2', 'S3', 'RDS')
            month: Month name (e.g., 'Jan', 'Feb', 'Mar')
        
        Returns:
            Dictionary with prediction details
        """
        # Validate inputs
        if region not in self.regions:
            raise ValueError(f"Invalid region. Choose from: {self.regions}")
        if service not in self.services:
            raise ValueError(f"Invalid service. Choose from: {self.services}")
        if month not in self.month_mapping:
            raise ValueError(f"Invalid month. Choose from: {list(self.month_mapping.keys())}")
        
        # Encode features
        region_encoded = self.region_encoder.transform([region])[0]
        service_encoded = self.service_encoder.transform([service])[0]
        month_number = self.month_mapping[month]
        
        # Create feature array
        X = np.array([[region_encoded, service_encoded, month_number]])
        
        # Predict
        predicted_cost = self.model.predict(X)[0]
        
        return {
            'region': region,
            'service': service,
            'month': month,
            'predicted_cost': round(predicted_cost, 2)
        }
    
    def predict_aggregated(self, service: str, month: str) -> Dict:
        """
        Predict aggregated cost across all regions for a service and month.
        
        Args:
            service: Service name (e.g., 'EC2', 'S3', 'RDS')
            month: Month name (e.g., 'Jan', 'Feb', 'Mar')
        
        Returns:
            Dictionary with aggregated prediction and breakdown by region
        """
        # Validate inputs
        if service not in self.services:
            raise ValueError(f"Invalid service. Choose from: {self.services}")
        if month not in self.month_mapping:
            raise ValueError(f"Invalid month. Choose from: {list(self.month_mapping.keys())}")
        
        # Predict for each region
        regional_predictions = {}
        total_cost = 0
        
        for region in self.regions:
            result = self.predict_single(region, service, month)
            regional_predictions[region] = result['predicted_cost']
            total_cost += result['predicted_cost']
        
        return {
            'service': service,
            'month': month,
            'total_cost_all_regions': round(total_cost, 2),
            'breakdown_by_region': regional_predictions
        }
    
    def predict_batch(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make batch predictions from a DataFrame.
        
        Args:
            predictions_df: DataFrame with columns ['Region', 'Service', 'Month']
                          Region can be None for aggregated predictions
        
        Returns:
            DataFrame with predictions
        """
        results = []
        
        for _, row in predictions_df.iterrows():
            if pd.isna(row.get('Region')) or row.get('Region') == '' or row.get('Region') is None:
                # Aggregated prediction
                result = self.predict_aggregated(row['Service'], row['Month'])
                results.append({
                    'Region': 'All Regions',
                    'Service': result['service'],
                    'Month': result['month'],
                    'Predicted_Cost': result['total_cost_all_regions']
                })
            else:
                # Single region prediction
                result = self.predict_single(row['Region'], row['Service'], row['Month'])
                results.append({
                    'Region': result['region'],
                    'Service': result['service'],
                    'Month': result['month'],
                    'Predicted_Cost': result['predicted_cost']
                })
        
        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = AWSCostPredictor('/mnt/user-data/outputs/aws_cost_prediction_model.pkl')
    
    print("="*70)
    print("AWS COST PREDICTION SYSTEM")
    print("="*70)
    
    # Example 1: Predict for specific region, service, and month
    print("\n📍 Example 1: Predict cost for US Region, EC2, March")
    print("-"*70)
    result1 = predictor.predict_single('US Region', 'EC2', 'Mar')
    print(f"Region:  {result1['region']}")
    print(f"Service: {result1['service']}")
    print(f"Month:   {result1['month']}")
    print(f"💰 Predicted Cost: ${result1['predicted_cost']:,.2f}")
    
    # Example 2: Predict aggregated cost across all regions
    print("\n🌍 Example 2: Predict aggregated S3 cost for June (all regions)")
    print("-"*70)
    result2 = predictor.predict_aggregated('S3', 'Jun')
    print(f"Service: {result2['service']}")
    print(f"Month:   {result2['month']}")
    print(f"💰 Total Cost (All Regions): ${result2['total_cost_all_regions']:,.2f}")
    print(f"\nBreakdown:")
    for region, cost in result2['breakdown_by_region'].items():
        print(f"  {region:15s}: ${cost:,.2f}")
    
    # Example 3: Multiple predictions
    print("\n📊 Example 3: Batch predictions")
    print("-"*70)
    predictions_to_make = pd.DataFrame([
        {'Region': 'US Region', 'Service': 'Lambda', 'Month': 'Sep'},
        {'Region': 'EU Region', 'Service': 'RDS', 'Month': 'Jul'},
        {'Region': None, 'Service': 'DynamoDB', 'Month': 'Oct'},  # Aggregated
        {'Region': 'APAC Region', 'Service': 'EKS', 'Month': 'May'},
    ])
    
    batch_results = predictor.predict_batch(predictions_to_make)
    print(batch_results.to_string(index=False))
    
    print("\n" + "="*70)
    print("Available Services:", ", ".join(predictor.services))
    print("Available Regions:", ", ".join(predictor.regions))
    print("Available Months:", ", ".join(predictor.month_mapping.keys()))
    print("="*70)
