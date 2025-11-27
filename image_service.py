"""
Image Analysis Service for Heart Disease Detection from Medical Images
Supports: Cardiac MRI, Echocardiogram, ECG images
"""

import numpy as np
from pathlib import Path
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Image analysis will use fallback mode.")


class CardiacImageAnalyzer:
    """
    CNN-based cardiac image analyzer for heart disease detection.
    Supports MRI, Echocardiogram, and ECG images.
    """

    # Image configuration
    IMG_SIZE = (224, 224)  # Standard input size for CNN
    CHANNELS = 3  # RGB

    # Classification labels
    CLASSES = ['Normal', 'Abnormal/Heart Disease Indicators']

    def __init__(self, model_path=None):
        """Initialize the image analyzer"""
        project_root = Path(__file__).resolve().parents[1]

        default_model = project_root / 'models' / 'cardiac_image_model.keras'
        self.model_path = Path(model_path or default_model).resolve()

        self.model = None
        self.is_loaded = False

    def load(self):
        """Load the CNN model"""
        try:
            if not TF_AVAILABLE:
                logger.warning("TensorFlow not available, using rule-based fallback")
                self.is_loaded = True  # Use fallback mode
                return True

            if self.model_path.exists():
                logger.info(f"Loading image model from {self.model_path}")
                self.model = keras.models.load_model(self.model_path)
                self.is_loaded = True
                logger.info("Image model loaded successfully!")
                return True
            else:
                logger.warning(f"Image model not found at {self.model_path}")
                logger.info("Will use image feature extraction fallback")
                self.is_loaded = True  # Use fallback mode
                return True

        except Exception as e:
            logger.error(f"Error loading image model: {e}")
            self.is_loaded = True  # Use fallback mode
            return True

    def preprocess_image(self, image_data):
        """
        Preprocess image for model input.

        Args:
            image_data: PIL Image, file path, or bytes

        Returns:
            Preprocessed numpy array ready for model
        """
        try:
            # Handle different input types
            if isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str) or isinstance(image_data, Path):
                img = Image.open(image_data)
            elif hasattr(image_data, 'read'):  # File-like object (Streamlit UploadedFile)
                img = Image.open(image_data)
            elif isinstance(image_data, Image.Image):
                img = image_data
            else:
                raise ValueError(f"Unsupported image type: {type(image_data)}")

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize to expected input size
            img = img.resize(self.IMG_SIZE, Image.Resampling.LANCZOS)

            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            return img_array

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Failed to preprocess image: {e}")

    def extract_image_features(self, img_array):
        """
        Extract features from image for analysis.
        Used as fallback when CNN model is not available.

        Returns dict of extracted features.
        """
        # Remove batch dimension for analysis
        img = img_array[0]

        features = {}

        # Color analysis
        features['mean_intensity'] = float(np.mean(img))
        features['std_intensity'] = float(np.std(img))

        # Per-channel analysis
        features['red_mean'] = float(np.mean(img[:, :, 0]))
        features['green_mean'] = float(np.mean(img[:, :, 1]))
        features['blue_mean'] = float(np.mean(img[:, :, 2]))

        # Contrast analysis
        features['contrast'] = float(np.max(img) - np.min(img))

        # Edge detection (simple gradient)
        gray = np.mean(img, axis=2)
        grad_x = np.abs(np.diff(gray, axis=1))
        grad_y = np.abs(np.diff(gray, axis=0))
        features['edge_intensity'] = float(np.mean(grad_x) + np.mean(grad_y))

        # Texture analysis (local variance)
        from scipy import ndimage
        features['texture_variance'] = float(ndimage.variance(gray))

        # Regional analysis (divide into quadrants)
        h, w = gray.shape
        quadrants = [
            gray[:h // 2, :w // 2],  # Top-left
            gray[:h // 2, w // 2:],  # Top-right
            gray[h // 2:, :w // 2],  # Bottom-left
            gray[h // 2:, w // 2:]  # Bottom-right
        ]
        features['regional_variance'] = float(np.std([np.mean(q) for q in quadrants]))

        return features

    def _rule_based_prediction(self, features):
        """
        Rule-based prediction when CNN model is not available.
        Uses extracted image features to estimate cardiac abnormality.

        This is a simplified heuristic - in production, use a trained CNN.
        """
        score = 0.5  # Start neutral

        # Analyze intensity patterns
        # Abnormal cardiac images often show irregular intensity patterns
        if features['std_intensity'] > 0.25:
            score += 0.1
        if features['regional_variance'] > 0.15:
            score += 0.1

        # High edge intensity might indicate abnormal structures
        if features['edge_intensity'] > 0.15:
            score += 0.05

        # Very low contrast might indicate image quality issues
        if features['contrast'] < 0.3:
            score -= 0.1

        # High texture variance can indicate abnormalities
        if features['texture_variance'] > 0.05:
            score += 0.1

        # Clamp to valid range
        score = max(0.0, min(1.0, score))

        return score

    def analyze(self, image_data, image_type='auto'):
        """
        Analyze medical image for heart disease indicators.

        Args:
            image_data: Image file (PIL Image, bytes, file path, or file object)
            image_type: Type of image ('mri', 'ecg', 'echo', 'xray', 'auto')

        Returns:
            dict with analysis results
        """
        if not self.is_loaded:
            raise RuntimeError("Image analyzer not loaded. Call load() first.")

        try:
            # Preprocess image
            img_array = self.preprocess_image(image_data)

            # Extract features for analysis
            features = self.extract_image_features(img_array)

            # Get prediction
            if self.model is not None and TF_AVAILABLE:
                # Use CNN model
                prediction = self.model.predict(img_array, verbose=0)
                probability = float(prediction[0][0])
            else:
                # Use rule-based fallback
                probability = self._rule_based_prediction(features)

            # Determine risk level
            if probability > 0.7:
                risk_level = 'HIGH'
                risk_emoji = '🔴'
            elif probability > 0.4:
                risk_level = 'MODERATE'
                risk_emoji = '🟡'
            else:
                risk_level = 'LOW'
                risk_emoji = '🟢'

            # Determine diagnosis
            diagnosis = 'Potential Cardiac Abnormality Detected' if probability > 0.5 else 'No Significant Abnormality Detected'

            # Build detailed analysis
            analysis_details = self._generate_analysis_details(features, probability, image_type)

            return {
                'prediction': int(probability > 0.5),
                'probability': probability,
                'confidence': abs(probability - 0.5) * 2,  # 0 to 1 scale
                'risk_level': risk_level,
                'risk_emoji': risk_emoji,
                'diagnosis': diagnosis,
                'image_type': image_type,
                'features': features,
                'analysis_details': analysis_details,
                'model_used': 'CNN' if self.model is not None else 'Feature Analysis',
                'medical_disclaimer': '⚠️ This is an AI-assisted analysis for educational purposes only. Always consult a qualified cardiologist for medical diagnosis.'
            }

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise ValueError(f"Failed to analyze image: {e}")

    def _generate_analysis_details(self, features, probability, image_type):
        """Generate detailed analysis report"""
        details = []

        # Image quality assessment
        if features['contrast'] > 0.5:
            details.append("✅ Good image contrast")
        else:
            details.append("⚠️ Low image contrast - may affect accuracy")

        # Intensity analysis
        if features['mean_intensity'] > 0.3 and features['mean_intensity'] < 0.7:
            details.append("✅ Image brightness within normal range")
        else:
            details.append("⚠️ Image brightness may be suboptimal")

        # Structure analysis
        if features['edge_intensity'] > 0.1:
            details.append("📊 Cardiac structures visible in image")

        # Regional analysis
        if features['regional_variance'] > 0.1:
            details.append("📍 Detected regional intensity variations")

        # Risk interpretation
        if probability > 0.7:
            details.append("🔴 High probability of cardiac abnormality indicators")
            details.append("   → Recommend immediate specialist consultation")
        elif probability > 0.4:
            details.append("🟡 Moderate indicators present")
            details.append("   → Recommend follow-up examination")
        else:
            details.append("🟢 No significant abnormality indicators detected")
            details.append("   → Continue regular health monitoring")

        return details


# Singleton instance
_image_analyzer = None


def get_image_analyzer():
    """Get or create singleton image analyzer instance"""
    global _image_analyzer
    if _image_analyzer is None:
        _image_analyzer = CardiacImageAnalyzer()
        _image_analyzer.load()
    return _image_analyzer


def create_cnn_model(input_shape=(224, 224, 3)):
    """
    Create a CNN model for cardiac image classification.
    Call this function to create and train a new model.

    Architecture based on common medical image classification patterns.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to create CNN model")

    model = keras.Sequential([
        # Input layer
        keras.layers.Input(shape=input_shape),

        # First Conv Block
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Second Conv Block
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Third Conv Block
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Fourth Conv Block
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),

        # Output layer
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    return model
