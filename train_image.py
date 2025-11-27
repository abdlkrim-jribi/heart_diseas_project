"""
Training script for cardiac image classification CNN model.

Usage:
    python train_image_model.py --data_dir /path/to/cardiac/images

Dataset structure expected:
    data_dir/
        train/
            normal/
                img1.jpg
                img2.jpg
            abnormal/
                img1.jpg
                img2.jpg
        validation/
            normal/
            abnormal/
"""

import argparse
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.error("TensorFlow is required for training. Install with: pip install tensorflow")


def create_data_generators(data_dir, img_size=(224, 224), batch_size=32):
    """Create training and validation data generators with augmentation"""

    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Use 20% for validation if no separate val folder
    )

    # Validation data (no augmentation, only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    data_path = Path(data_dir)

    # Check if we have separate train/validation folders
    if (data_path / 'train').exists():
        train_dir = data_path / 'train'
        val_dir = data_path / 'validation' if (data_path / 'validation').exists() else data_path / 'train'

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False,
            subset='validation' if val_dir == train_dir else None
        )
    else:
        # Use validation_split from single directory
        train_generator = train_datagen.flow_from_directory(
            data_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True,
            subset='training'
        )

        val_generator = val_datagen.flow_from_directory(
            data_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False,
            subset='validation'
        )

    return train_generator, val_generator


def train_model(data_dir, epochs=50, batch_size=32, model_save_path=None):
    """Train the cardiac image classification model"""

    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for training")

    # Import the model architecture
    from image_service import create_cnn_model

    # Set up paths
    project_root = Path(__file__).resolve().parents[1]
    if model_save_path is None:
        model_save_path = project_root / 'models' / 'cardiac_image_model.keras'

    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CARDIAC IMAGE MODEL TRAINING")
    logger.info("=" * 60)

    # Create data generators
    logger.info(f"Loading data from: {data_dir}")
    train_gen, val_gen = create_data_generators(data_dir, batch_size=batch_size)

    logger.info(f"Training samples: {train_gen.samples}")
    logger.info(f"Validation samples: {val_gen.samples}")
    logger.info(f"Classes: {train_gen.class_indices}")

    # Create model
    logger.info("\nCreating CNN model...")
    model = create_cnn_model()
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(model_save_path),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(project_root / 'logs' / 'image_model'),
            histogram_freq=1
        )
    ]

    # Train
    logger.info("\nStarting training...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(model_save_path)
    logger.info(f"\n✅ Model saved to: {model_save_path}")

    # Evaluate
    logger.info("\nFinal Evaluation:")
    val_loss, val_acc, val_auc = model.evaluate(val_gen, verbose=0)
    logger.info(f"  Validation Loss: {val_loss:.4f}")
    logger.info(f"  Validation Accuracy: {val_acc:.4f}")
    logger.info(f"  Validation AUC: {val_auc:.4f}")

    return model, history


def create_synthetic_dataset(output_dir, num_samples=100):
    """
    Create a synthetic dataset for testing/demo purposes.
    In production, use real cardiac images from medical datasets.
    """
    from PIL import Image
    import random

    output_path = Path(output_dir)

    # Create directory structure
    for split in ['train', 'validation']:
        for cls in ['normal', 'abnormal']:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating synthetic dataset in {output_path}")

    # Generate synthetic images
    for split in ['train', 'validation']:
        n_samples = num_samples if split == 'train' else num_samples // 5

        for cls in ['normal', 'abnormal']:
            for i in range(n_samples):
                # Create synthetic cardiac-like image
                img = np.zeros((224, 224, 3), dtype=np.uint8)

                # Background
                if cls == 'normal':
                    # More uniform, healthy appearance
                    base_color = random.randint(40, 80)
                    img[:, :] = [base_color, base_color, base_color]

                    # Add smooth heart shape
                    center = (112, 112)
                    for y in range(224):
                        for x in range(224):
                            dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                            if dist < 60:
                                intensity = int(base_color + 40 * (1 - dist / 60))
                                img[y, x] = [intensity, intensity - 10, intensity - 10]
                else:
                    # More irregular, abnormal appearance
                    base_color = random.randint(30, 60)
                    img[:, :] = [base_color, base_color, base_color]

                    # Add irregular patches (simulating abnormalities)
                    num_patches = random.randint(3, 7)
                    for _ in range(num_patches):
                        px, py = random.randint(20, 200), random.randint(20, 200)
                        size = random.randint(10, 40)
                        intensity = random.randint(80, 150)
                        img[max(0, py - size):min(224, py + size),
                        max(0, px - size):min(224, px + size)] = [intensity, intensity - 20, intensity - 10]

                # Add some noise
                noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                # Save image
                pil_img = Image.fromarray(img)
                pil_img.save(output_path / split / cls / f'{cls}_{i:04d}.png')

    logger.info(
        f"✅ Synthetic dataset created with {num_samples * 2} training and {num_samples // 5 * 2} validation images")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train cardiac image classification model')
    parser.add_argument('--data_dir', type=str, help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--create_synthetic', action='store_true', help='Create synthetic dataset for testing')
    parser.add_argument('--output_dir', type=str, default='data/cardiac_images',
                        help='Output directory for synthetic data')

    args = parser.parse_args()

    if args.create_synthetic:
        # Create synthetic dataset for demo
        data_path = create_synthetic_dataset(args.output_dir)
        print(f"\nTo train on this data, run:")
        print(f"  python train_image_model.py --data_dir {data_path}")
    elif args.data_dir:
        # Train on provided data
        train_model(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
    else:
        print("Usage:")
        print("  Create synthetic data: python train_image_model.py --create_synthetic")
        print("  Train model: python train_image_model.py --data_dir /path/to/data")
