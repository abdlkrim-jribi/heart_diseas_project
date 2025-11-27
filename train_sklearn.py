import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib
from ucimlrepo import fetch_ucirepo  # Downloads heart disease data

# Load & preprocess (exactly as notebook)
heartdisease = fetch_ucirepo(id=45)
X = heartdisease.data.features[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                               'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = (heartdisease.data.targets['num'] > 0).astype(int)  # Binary 0/1

X = X.fillna(X.median())  # Handle ca/thal NaNs
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)

# Eval (89% acc)
print(f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")

# Save to models/
joblib.dump(model, 'models/heart_model_gnb.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Saved models/heart_model_gnb.pkl & scaler.pkl")
