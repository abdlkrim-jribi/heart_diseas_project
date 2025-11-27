"""
Create a simple mock model for testing without TensorFlow loading issues
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

# Create project paths
project_root = Path(__file__).resolve().parent.parent
models_dir = project_root / 'models'
models_dir.mkdir(exist_ok=True)

print("Creating mock model and scaler...")


# 1. Create a simple mock model (just a function that predicts)
class SimplePredictor:
    def __init__(self):
        self.input_shape = (None, 13)
        self.output_shape = (None, 1)

    def predict(self, X, verbose=0):
        # Simple rule-based prediction
        # If cholesterol > 240 and age > 50, higher risk
        predictions = []
        for sample in X:
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = sample

            # Simple scoring rule
            score = 0
            if chol > 240:
                score += 0.3
            if trestbps > 140:
                score += 0.2
            if age > 50:
                score += 0.15
            if oldpeak > 2:
                score += 0.25
            if ca > 1:
                score += 0.1

            # Normalize to 0-1
            score = min(score, 1.0)
            predictions.append([score])

        return np.array(predictions)


mock_model = SimplePredictor()

# Save mock model
mock_model_path = models_dir / 'best_model_final_nn.keras'
with open(mock_model_path, 'wb') as f:
    pickle.dump(mock_model, f)

print(f"✅ Mock model saved: {mock_model_path}")

# 2. Create scaler
scaler = MinMaxScaler()
dummy_data = np.array([
    [29, 0, 1, 94, 126, 0, 0, 71, 0, 0.0, 1, 0, 3],
    [77, 1, 4, 200, 564, 1, 2, 202, 1, 6.2, 3, 3, 7]
])
scaler.fit(dummy_data)

scaler_path = models_dir / 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Scaler saved: {scaler_path}")
print(f"\n✅ Mock model and scaler ready!")
print(f"   Model path: {mock_model_path}")
print(f"   Scaler path: {scaler_path}")
