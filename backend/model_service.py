"""
Heart Disease Predictor - Sklearn Version (No TensorFlow)
Compatible with your trained GaussianNB model from train_sklearn.py
"""

import joblib
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeartDiseasePredictor:
    """Heart disease prediction using Gaussian Naive Bayes"""

    FEATURE_NAMES = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    FEATURE_CONFIG = {
        'age': {'range': (29, 77), 'type': int, 'label': 'Age (years)'},
        'sex': {'range': (0, 1), 'type': int, 'label': 'Sex (0=Female, 1=Male)'},
        'cp': {'range': (1, 4), 'type': int, 'label': 'Chest Pain Type'},
        'trestbps': {'range': (94, 200), 'type': int, 'label': 'Resting Blood Pressure (mm Hg)'},
        'chol': {'range': (126, 564), 'type': int, 'label': 'Serum Cholesterol (mg/dl)'},
        'fbs': {'range': (0, 1), 'type': int, 'label': 'Fasting Blood Sugar > 120 mg/dl'},
        'restecg': {'range': (0, 2), 'type': int, 'label': 'Resting ECG'},
        'thalach': {'range': (71, 202), 'type': int, 'label': 'Maximum Heart Rate'},
        'exang': {'range': (0, 1), 'type': int, 'label': 'Exercise Induced Angina'},
        'oldpeak': {'range': (0.0, 6.2), 'type': float, 'label': 'ST Depression'},
        'slope': {'range': (1, 3), 'type': int, 'label': 'Slope of Peak Exercise ST'},
        'ca': {'range': (0, 3), 'type': int, 'label': 'Number of Major Vessels'},
        'thal': {'range': (3, 7), 'type': int, 'label': 'Thalassemia Type'}
    }

    def __init__(self, model_path=None, scaler_path=None):
        """Initialize predictor with sklearn model paths"""
        project_root = Path(__file__).resolve().parents[1]

        default_model = project_root / 'models' / 'heart_model_gnb.pkl'
        default_scaler = project_root / 'models' / 'scaler.pkl'

        self.model_path = Path(model_path or default_model).resolve()
        self.scaler_path = Path(scaler_path or default_scaler).resolve()

        self.model = None
        self.scaler = None
        self.is_loaded = False

    def load(self):
        """Load sklearn model and scaler"""
        try:
            print(f"\n{'=' * 60}")
            print("LOADING MODEL AND SCALER")
            print(f"{'=' * 60}")

            if not self.model_path.exists():
                print(f"❌ Model not found: {self.model_path}")
                print(f"   Run: python train_sklearn.py")
                self.is_loaded = False
                return False

            if not self.scaler_path.exists():
                print(f"❌ Scaler not found: {self.scaler_path}")
                print(f"   Run: python train_sklearn.py")
                self.is_loaded = False
                return False

            print(f"✅ Model: {self.model_path.name}")
            print(f"✅ Scaler: {self.scaler_path.name}")

            # Load scaler
            print(f"\n🔄 Loading scaler...")
            self.scaler = joblib.load(self.scaler_path)
            print(f"✅ Scaler loaded ({self.scaler.n_features_in_} features)")

            # Load model
            print(f"\n🔄 Loading model...")
            self.model = joblib.load(self.model_path)
            print(f"✅ Sklearn GaussianNB loaded!")

            # Test prediction
            print(f"\n🧪 Testing...")
            test = np.array([[50, 1, 3, 130, 250, 0, 1, 140, 0, 1.5, 2, 0, 3]])
            test_scaled = self.scaler.transform(test)
            pred = self.model.predict_proba(test_scaled)[0][1]
            print(f"✅ Test prediction: {pred:.4f}")

            self.is_loaded = True

            print(f"\n{'=' * 60}")
            print("✅ READY! is_loaded = True")
            print(f"{'=' * 60}\n")

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
            return False

    def validate_features(self, features):
        """Validate input features"""
        if len(features) != 13:
            return False, f"Expected 13 features, got {len(features)}"

        for name in self.FEATURE_NAMES:
            if name not in features:
                return False, f"Missing feature: {name}"

        return True, None

    def preprocess(self, features):
        """Convert feature dict to scaled array"""
        arr = np.array([[features[name] for name in self.FEATURE_NAMES]])
        return self.scaler.transform(arr)

    def predict(self, features):
        """Make heart disease prediction"""
        if not self.is_loaded or self.model is None or self.scaler is None:
            error_msg = f"Model not loaded! (is_loaded={self.is_loaded})"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

        # Validate
        valid, err = self.validate_features(features)
        if not valid:
            raise ValueError(err)

        # Predict
        try:
            scaled = self.preprocess(features)
            prob = float(self.model.predict_proba(scaled)[0][1])

            # Risk levels
            if prob > 0.7:
                risk, emoji = 'HIGH', '🔴'
            elif prob > 0.4:
                risk, emoji = 'MODERATE', '🟡'
            else:
                risk, emoji = 'LOW', '🟢'

            return {
                'prediction': int(prob > 0.5),
                'probability': prob,
                'risk_level': risk,
                'risk_emoji': emoji,
                'diagnosis': 'Heart Disease' if prob > 0.5 else 'No Heart Disease',
                'medical_disclaimer': '⚠️ Educational purposes only. Consult a healthcare professional.'
            }
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            raise


# Singleton instance
_predictor = None


def get_predictor():
    """Get or create predictor singleton"""
    global _predictor

    if _predictor is None:
        print("Creating new predictor instance...")
        _predictor = HeartDiseasePredictor()
        success = _predictor.load()
        if not success:
            raise RuntimeError("Failed to load model! Run: python train_sklearn.py")
        print(f"✅ Predictor ready (is_loaded={_predictor.is_loaded})")
    else:
        print(f"Using existing predictor (is_loaded={_predictor.is_loaded})")

    return _predictor
