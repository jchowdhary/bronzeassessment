"""
Interactive AWS Cost Prediction Demo
-------------------------------------
Easy-to-use script for testing the prediction model
"""

import pickle
import numpy as np
import pandas as pd

def load_model():
    """Load the trained model and encoders"""
    with open('/mnt/user-data/outputs/aws_cost_prediction_model.pkl', 'rb') as f:
        return pickle.load(f)

def predict_cost(model_data, region, service, month):
    """Make a single prediction"""
    model = model_data['model']
    region_encoder = model_data['region_encoder']
    service_encoder = model_data['service_encoder']
    month_mapping = model_data['month_mapping']
    
    # Encode features
    region_encoded = region_encoder.transform([region])[0]
    service_encoded = service_encoder.transform([service])[0]
    month_number = month_mapping[month]
    
    # Predict
    X = np.array([[region_encoded, service_encoded, month_number]])
    predicted_cost = model.predict(X)[0]
    
    return predicted_cost

def predict_aggregated(model_data, service, month):
    """Predict aggregated cost across all regions"""
    regions = model_data['regions']
    costs = {}
    total = 0
    
    for region in regions:
        cost = predict_cost(model_data, region, service, month)
        costs[region] = cost
        total += cost
    
    return total, costs

def main():
    print("\n" + "="*70)
    print("🚀 AWS COST PREDICTION - INTERACTIVE DEMO")
    print("="*70)
    
    # Load model
    print("\n📦 Loading model...")
    model_data = load_model()
    print("✅ Model loaded successfully!")
    
    print(f"\n📋 Available Options:")
    print(f"   Regions: {', '.join(model_data['regions'])}")
    print(f"   Services: {', '.join(model_data['services'])}")
    print(f"   Months: {', '.join(model_data['month_mapping'].keys())}")
    
    # Demo predictions
    print("\n" + "="*70)
    print("📍 TEST 1: Specific Region Prediction")
    print("="*70)
    
    test_cases_single = [
        ('US Region', 'EC2', 'Jan'),
        ('EU Region', 'S3', 'Jun'),
        ('APAC Region', 'Lambda', 'Dec'),
        ('US Region', 'RDS', 'Sep'),
    ]
    
    for region, service, month in test_cases_single:
        cost = predict_cost(model_data, region, service, month)
        print(f"💰 {region:15s} | {service:12s} | {month:3s} → ${cost:>10,.2f}")
    
    # Aggregated predictions
    print("\n" + "="*70)
    print("🌍 TEST 2: Aggregated Predictions (All Regions)")
    print("="*70)
    
    test_cases_aggregated = [
        ('EC2', 'Mar'),
        ('S3', 'Jul'),
        ('DynamoDB', 'Oct'),
    ]
    
    for service, month in test_cases_aggregated:
        total, breakdown = predict_aggregated(model_data, service, month)
        print(f"\n💰 {service:12s} | {month:3s} → Total: ${total:,.2f}")
        for region, cost in breakdown.items():
            print(f"   └─ {region:15s}: ${cost:>10,.2f}")
    
    # Custom prediction
    print("\n" + "="*70)
    print("🎯 CUSTOM PREDICTION")
    print("="*70)
    
    print("\nTry your own prediction!")
    print("\nExample inputs:")
    print("  Region: US Region (or leave blank for all regions)")
    print("  Service: EC2")
    print("  Month: Jan")
    
    try:
        region_input = input("\nEnter Region (or press Enter for all regions): ").strip()
        service_input = input("Enter Service: ").strip()
        month_input = input("Enter Month: ").strip()
        
        if not service_input or not month_input:
            print("❌ Service and Month are required!")
        elif not region_input:
            # Aggregated prediction
            total, breakdown = predict_aggregated(model_data, service_input, month_input)
            print(f"\n✅ Predicted Total Cost (All Regions): ${total:,.2f}")
            print("\nBreakdown:")
            for region, cost in breakdown.items():
                print(f"   {region:15s}: ${cost:>10,.2f}")
        else:
            # Single region prediction
            cost = predict_cost(model_data, region_input, service_input, month_input)
            print(f"\n✅ Predicted Cost: ${cost:,.2f}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check your inputs match the available options.")
    
    print("\n" + "="*70)
    print("✨ Demo completed!")
    print("="*70)

if __name__ == "__main__":
    main()
