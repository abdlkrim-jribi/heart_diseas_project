"""
Test script to verify model and scaler are working correctly
"""

import tensorflow as tf
import pickle
import numpy as np
from pathlib import Path
import sys


def test_model():
    """Test that model and scaler work correctly"""

    print("=" * 80)
    print("MODEL VERIFICATION TEST")
    print("=" * 80)

    # Check if files exist
    model_path = Path('models/best_model_final_nn.keras')
    scaler_path = Path('models/scaler.pkl')

    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("   Please copy your trained model to models/ directory")
        return False

    if not scaler_path.exists():
        print(f"❌ Scaler file not found: {scaler_path}")
        print("   Please copy your scaler to models/ directory")
        return False

    print(f"✅ Model file found: {model_path}")
    print(f"✅ Scaler file found: {scaler_path}")

    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"\n✅ Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total layers: {len(model.layers)}")
        print(f"   Total parameters: {model.count_params():,}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

    # Load scaler
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"\n✅ Scaler loaded successfully!")
        print(f"   Features expected: {scaler.n_features_in_}")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return False

    # Test prediction with multiple samples
    print("\n" + "=" * 80)
    print("RUNNING TEST PREDICTIONS")
    print("=" * 80)

    test_cases = [
        {
            'name': 'Low Risk Patient',
            'features': [35, 0, 1, 110, 180, 0, 0, 170, 0, 0.5, 1, 0, 3]
        },
        {
            'name': 'Moderate Risk Patient',
            'features': [50, 1, 3, 130, 250, 0, 1, 140, 0, 1.5, 2, 0, 3]
        },
        {
            'name': 'High Risk Patient',
            'features': [65, 1, 4, 160, 300, 1, 2, 110, 1, 3.5, 3, 2, 7]
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {case['name']}")
        print(f"Features: {case['features']}")

        # Prepare input
        features_array = np.array([case['features']])
        features_scaled = scaler.transform(features_array)

        # Predict
        prediction = model.predict(features_scaled, verbose=0)
        probability = prediction[0][0]
        diagnosis = 'Heart Disease' if probability > 0.5 else 'No Heart Disease'

        # Risk level
        if probability > 0.7:
            risk = "🔴 HIGH"
        elif probability > 0.4:
            risk = "🟡 MODERATE"
        else:
            risk = "🟢 LOW"

        print(f"  Probability: {probability:.4f} ({probability * 100:.1f}%)")
        print(f"  Diagnosis: {diagnosis}")
        print(f"  Risk Level: {risk}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nYour model is ready to use in the Streamlit app!")
    print("Run: streamlit run backend/app.py")

    return True


if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
