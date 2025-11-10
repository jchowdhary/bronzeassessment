# 🚀 AWS Cost Prediction System

## 📊 Model Performance

**Winner: Gradient Boosting Regressor**
- ✅ **R² Score: 99.74%** - The model explains 99.74% of cost variance
- ✅ **MAE: $98.98** - Average prediction error is only $98.98
- ✅ **RMSE: $196.46** - Root mean square error of $196.46

### Model Comparison Results

| Model               | Train R²  | Test R²   | MAE      | RMSE     |
|---------------------|-----------|-----------|----------|----------|
| Gradient Boosting   | 0.9999    | 0.9974    | $98.98   | $196.46  |
| Decision Tree       | 1.0000    | 0.9959    | $152.22  | $246.20  |
| Random Forest       | 0.9996    | 0.9945    | $148.58  | $283.38  |
| Linear Regression   | 0.0339    | 0.0200    | $2,857   | $3,797   |

**Why Gradient Boosting?**
- Best test performance (R² = 0.9974)
- Lowest prediction error (MAE = $98.98)
- Handles non-linear relationships excellently
- Robust against overfitting with proper hyperparameters

---

## 📁 Files Included

1. **aws_cost_prediction_model.pkl** - Trained Gradient Boosting model with encoders
2. **aws_cost_predictor.py** - Python script for making predictions
3. **model_performance_analysis.png** - Visual analysis of model performance
4. **feature_importance.png** - Feature importance chart
5. **AWSFinops_All_Regions.csv** - Combined dataset (long format)
6. **AWSFinops_Long_Format.csv** - Reshaped data for modeling

---

## 🎯 Usage Guide

### Option 1: Using the Python Script

```python
from aws_cost_predictor import AWSCostPredictor

# Initialize predictor
predictor = AWSCostPredictor('aws_cost_prediction_model.pkl')

# Case 1: Predict cost for specific region, service, and month
result = predictor.predict_single('US Region', 'EC2', 'Mar')
print(f"Predicted Cost: ${result['predicted_cost']:,.2f}")

# Case 2: Predict aggregated cost across all regions
result = predictor.predict_aggregated('S3', 'Jun')
print(f"Total Cost (All Regions): ${result['total_cost_all_regions']:,.2f}")
print(f"Breakdown: {result['breakdown_by_region']}")
```

### Option 2: Quick Predictions in Python

```python
import pickle
import numpy as np

# Load model
with open('aws_cost_prediction_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
region_encoder = model_data['region_encoder']
service_encoder = model_data['service_encoder']
month_mapping = model_data['month_mapping']

# Make prediction
region_encoded = region_encoder.transform(['US Region'])[0]
service_encoded = service_encoder.transform(['EC2'])[0]
month_number = month_mapping['Mar']

X = np.array([[region_encoded, service_encoded, month_number]])
predicted_cost = model.predict(X)[0]

print(f"Predicted Cost: ${predicted_cost:,.2f}")
```

---

## 📋 Supported Values

### Regions (3)
- APAC Region
- EU Region
- US Region

### Services (15)
- Athena
- CloudFront
- CloudWatch
- DynamoDB
- EC2
- ECS
- EKS
- ELB
- Firehose
- Glue
- Lambda
- RDS
- Redshift
- S3
- VPC

### Months (12)
- Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec

---

## 🔍 Example Predictions

### Example 1: Single Region Prediction
```python
predictor.predict_single('US Region', 'EC2', 'Mar')
```
**Output:**
```
{
    'region': 'US Region',
    'service': 'EC2',
    'month': 'Mar',
    'predicted_cost': 13977.01
}
```

### Example 2: Aggregated Prediction (All Regions)
```python
predictor.predict_aggregated('S3', 'Jun')
```
**Output:**
```
{
    'service': 'S3',
    'month': 'Jun',
    'total_cost_all_regions': 9620.92,
    'breakdown_by_region': {
        'APAC Region': 2581.76,
        'EU Region': 3145.42,
        'US Region': 3893.74
    }
}
```

### Example 3: Batch Predictions
```python
import pandas as pd

predictions = pd.DataFrame([
    {'Region': 'US Region', 'Service': 'Lambda', 'Month': 'Sep'},
    {'Region': 'EU Region', 'Service': 'RDS', 'Month': 'Jul'},
    {'Region': None, 'Service': 'DynamoDB', 'Month': 'Oct'},  # Aggregated
])

results = predictor.predict_batch(predictions)
print(results)
```

---

## 📊 Model Features

The model uses **3 features** to predict costs:

1. **Region** (categorical) - Encoded as 0, 1, 2
2. **Service** (categorical) - Encoded as 0-14
3. **Month** (temporal) - Numeric value 1-12

**Feature Importance:**
- Service: ~85% (Most important)
- Region: ~10%
- Month: ~5%

This makes sense as different services have vastly different cost structures, while regional and temporal variations are secondary.

---

## 🎨 Visualizations

### 1. Model Performance Analysis
Shows:
- Actual vs Predicted scatter plot
- Prediction error distribution
- Average cost by service
- Cost trends over months

### 2. Feature Importance
Bar chart showing which features contribute most to predictions.

---

## 💡 Key Insights

1. **Service Type** is the primary cost driver (85% importance)
2. **EC2** and **RDS** are the most expensive services
3. Costs show a **gradual upward trend** over months
4. The model achieves **near-perfect accuracy** with R² = 0.9974
5. Average prediction error is only **$98.98**

---

## 🚨 Error Handling

The predictor includes validation:

```python
# Invalid region
try:
    predictor.predict_single('Invalid Region', 'EC2', 'Jan')
except ValueError as e:
    print(e)  # "Invalid region. Choose from: ['APAC Region', 'EU Region', 'US Region']"
```

---

## 📈 Future Enhancements

Potential improvements:
1. Add support for cost forecasting (future months)
2. Include trend analysis and seasonality
3. Add confidence intervals for predictions
4. Support for additional AWS services
5. Integration with AWS Cost Explorer API

---

## 📞 Support

For questions or issues:
1. Check the example code in `aws_cost_predictor.py`
2. Review the visualizations for model behavior
3. Validate your inputs against supported values

---

## 🎯 Quick Start

```bash
# Run the demo
python aws_cost_predictor.py

# Use in your code
from aws_cost_predictor import AWSCostPredictor
predictor = AWSCostPredictor('aws_cost_prediction_model.pkl')
result = predictor.predict_single('US Region', 'EC2', 'Jan')
print(f"Cost: ${result['predicted_cost']:,.2f}")
```

---

**Model Created:** November 2025  
**Accuracy:** 99.74% R²  
**Algorithm:** Gradient Boosting Regressor  
**Training Data:** 450 records (3 regions × 15 services × 10 months)
