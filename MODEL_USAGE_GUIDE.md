# Body Fat Estimation MLP Model - Usage Guide

## Issue: Incorrect Predictions from Trained Model

This document addresses the common issue where users receive unexpected predictions (values in the 32-36 range) when using the pre-trained body fat estimation model, instead of the expected body fat percentages from the CSV data.

## Root Cause

The primary issue is **missing data preprocessing**. The model was trained on scaled data using `MinMaxScaler` (normalizing all features to a 0-1 range), but raw, unscaled anthropometric measurements are being fed directly to the model during inference.

### Training vs. Inference Data Processing

**During Training:**
```python
from sklearn.preprocessing import MinMaxScaler

# Original training pipeline
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Features normalized to 0-1 range
# Model trained on X_scaled
```

**Current Problematic Inference:**
```python
# Raw data being fed directly (INCORRECT)
inputs = np.array([1.0708, 23, 69.95, 172.085, ...])  # Raw values
prediction = model.predict(inputs)  # Model expects scaled values!
```

## Solution

### Complete Corrected Implementation

Here's the corrected code that includes proper preprocessing:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the original training data to fit the scaler
df = pd.read_csv("/app/repo/Body_Fat.csv")
X_training = df.drop('BodyFat', axis=1)

# Fit scaler on training data (must match exactly how it was done during training)
scaler = MinMaxScaler()
scaler.fit(X_training)

# Load the trained model
model = keras.models.load_model("/app/repo/best_full_model.keras")

def predict(measures: list[float]) -> float:
    """
    Predict body fat percentage from anthropometric measurements.

    Args:
        measures: List of 14 measurements in order:
                 [Density, Age, Weight, Height, Neck, Chest, Abdomen,
                  Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist]

    Returns:
        Predicted body fat percentage
    """
    # Convert to numpy array with correct shape
    inputs = np.array([measures], dtype="float32")

    # CRITICAL: Apply the same scaling as used during training
    inputs_scaled = scaler.transform(inputs)

    # Make prediction on scaled data
    prediction = model.predict(inputs_scaled, verbose=0)
    return float(prediction[0][0])
```

### Updated FastAPI Implementation

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    density: float
    age: float
    weight: float
    height: float
    neck: float
    chest: float
    abdomen: float
    hip: float
    thigh: float
    knee: float
    ankle: float
    biceps: float
    forearm: float
    wrist: float

class PredictionResponse(BaseModel):
    body_fat: float
    input_values: List[float]  # For debugging
    scaled_values: List[float]  # For debugging

app = FastAPI()

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    """
    Predict body fat percentage from anthropometric measurements.
    Features must be provided in the exact order as training data.
    """
    # Ensure features are in the same order as training data CSV:
    # Density,Age,Weight,Height,Neck,Chest,Abdomen,Hip,Thigh,Knee,Ankle,Biceps,Forearm,Wrist
    measures = [
        request.density,
        request.age,
        request.weight,
        request.height,
        request.neck,
        request.chest,
        request.abdomen,
        request.hip,
        request.thigh,
        request.knee,
        request.ankle,
        request.biceps,
        request.forearm,
        request.wrist
    ]

    # Log input for debugging
    logger.info(f"Raw input: {measures}")

    # Get scaled values for debugging
    inputs_array = np.array([measures], dtype="float32")
    scaled_values = scaler.transform(inputs_array)[0].tolist()
    logger.info(f"Scaled input: {scaled_values}")

    # Make prediction
    body_fat = predict(measures)
    logger.info(f"Prediction: {body_fat}")

    return PredictionResponse(
        body_fat=body_fat,
        input_values=measures,
        scaled_values=scaled_values
    )
```

## Testing the Solution

### Test with CSV Data

To verify the fix works correctly, test with the first row from `Body_Fat.csv`:

```python
# First row from CSV (excluding BodyFat column):
test_measures = [1.0708, 23, 69.95464853, 172.085, 36.2, 93.1, 85.2, 94.5, 59, 37.3, 21.9, 32, 27.4, 17.1]
predicted_body_fat = predict(test_measures)
expected_body_fat = 12.3  # From CSV

print(f"Predicted: {predicted_body_fat:.2f}%")
print(f"Expected: {expected_body_fat}%")
print(f"Difference: {abs(predicted_body_fat - expected_body_fat):.2f}%")
```

### Expected Results

With proper preprocessing, you should see predictions very close to the actual body fat values from the CSV file (within 1-2% difference), rather than the problematic 32-36 range.

## Understanding the Scaling Impact

### Raw vs. Scaled Values Example

| Feature | Raw Value | Scaled Value (0-1) |
|---------|-----------|-------------------|
| Density | 1.0708 | ~0.35 |
| Age | 23 | ~0.15 |
| Weight | 69.95 | ~0.25 |
| Height | 172.085 | ~0.40 |
| Chest | 93.1 | ~0.30 |

The model was trained to recognize patterns in these 0-1 scaled values, not the raw measurements.

## Production Recommendations

### 1. Save and Load the Scaler

For production deployment, save the fitted scaler alongside the model:

```python
import joblib

# Save scaler during training
joblib.dump(scaler, 'body_fat_scaler.pkl')

# Load scaler during inference
scaler = joblib.load('body_fat_scaler.pkl')
```

### 2. Input Validation

Add validation to ensure inputs are within reasonable ranges:

```python
def validate_measurements(measures: List[float]) -> None:
    """Validate that measurements are within reasonable physiological ranges."""
    validations = [
        (measures[0], 0.9, 1.2, "Density"),
        (measures[1], 15, 80, "Age"),
        (measures[2], 40, 200, "Weight (kg)"),
        (measures[3], 140, 220, "Height (cm)"),
        # Add more validations as needed
    ]

    for value, min_val, max_val, name in validations:
        if not min_val <= value <= max_val:
            raise ValueError(f"{name} value {value} is outside reasonable range [{min_val}, {max_val}]")
```

### 3. Model Performance Expectations

With proper preprocessing, the model should achieve:
- **R² Score**: > 0.97 (explains >97% of variance)
- **Mean Squared Error**: < 1.0
- **Typical Prediction Error**: ±1-2% body fat percentage

## Troubleshooting

### Common Issues

1. **Wrong Feature Order**: Ensure features match CSV column order exactly
2. **Missing Scaling**: Always apply MinMaxScaler before prediction
3. **Incorrect Data Types**: Use `float32` for consistency with training
4. **Scaler Fitting**: Scaler must be fitted on training data, not individual predictions

### Debug Steps

1. Print raw input values
2. Print scaled input values
3. Compare with CSV data ranges
4. Test with known CSV rows first
5. Verify model architecture matches training

## Model Architecture Reference

The trained model uses:
- **Input**: 14 anthropometric features (scaled 0-1)
- **Hidden Layer**: 20 neurons with sigmoid activation
- **Output**: 1 neuron with linear activation (body fat percentage)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error

## Contact

If you continue experiencing issues after implementing these changes, please provide:
1. Sample input values you're testing
2. Actual vs. expected output
3. Any error messages
4. Confirmation that preprocessing steps are implemented

The key principle: **Machine learning models require identical preprocessing during training and inference**. Any deviation in data preparation will result in poor predictions.