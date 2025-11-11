"""
Emotion Detection Model Module
==============================
Trains and uses a CNN model for emotion detection from images.
Uses FER-2013 dataset for training.
"""

import os
import requests
import pandas as pd
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Emotion labels
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MODEL_PATH = "model.h5"
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013"
DATASET_PATH = "fer2013.csv"

class EmotionDetector:
    """
    Emotion detection class using a trained Keras CNN model
    """

    def __init__(self):
        self.model = None
        self._loaded = False
        self.emotion_labels = EMOTION_LABELS
        self.db_path = "emotion_predictions.db"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for storing predictions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create predictions table if it doesn't exist (updated to include user_name)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_name TEXT,
                    predicted_emotion TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    image_path TEXT,
                    source TEXT DEFAULT 'unknown'
                )
            """)

            # Check if user_name column exists, add if not
            cursor.execute("PRAGMA table_info(predictions)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'user_name' not in columns:
                cursor.execute("ALTER TABLE predictions ADD COLUMN user_name TEXT")

            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")

    def download_dataset(self):
        """Download FER-2013 dataset if not present"""
        if os.path.exists(DATASET_PATH):
            logger.info("Dataset already exists")
            return

        logger.info("Downloading FER-2013 dataset...")
        try:
            response = requests.get(DATASET_URL, timeout=30)
            if response.status_code == 200 and 'text/csv' in response.headers.get('content-type', ''):
                with open(DATASET_PATH, 'wb') as f:
                    f.write(response.content)
                logger.info("Dataset downloaded successfully")
            else:
                logger.warning(f"Could not download dataset (status: {response.status_code}). Will use sample data.")
                # Create empty file to trigger sample data creation
                with open(DATASET_PATH, 'w') as f:
                    f.write("emotion,pixels\n")
        except Exception as e:
            logger.warning(f"Error downloading dataset: {e}. Will use sample data.")
            # Create empty file to trigger sample data creation
            with open(DATASET_PATH, 'w') as f:
                f.write("emotion,pixels\n")

    def load_and_preprocess_data(self):
        """Load and preprocess FER-2013 dataset"""
        self.download_dataset()

        try:
            data = pd.read_csv(DATASET_PATH)
            
            # Check if the CSV has the expected structure
            if 'pixels' not in data.columns or 'emotion' not in data.columns:
                logger.error("Invalid dataset format. Creating sample data for demonstration...")
                # Create a small sample dataset for demonstration
                return self._create_sample_data()
            
            # Filter by Usage if column exists, otherwise use all data
            if 'Usage' in data.columns:
                data = data[data['Usage'] == 'Training']
            
            # Limit dataset size for faster training (use first 10000 samples)
            data = data.head(10000)

            pixels = data['pixels'].tolist()
            emotions = data['emotion'].tolist()

            X = []
            for pixel_sequence in pixels:
                face = [int(pixel) for pixel in pixel_sequence.split(' ')]
                face = np.asarray(face).reshape(48, 48)
                face = face / 255.0  # Normalize
                X.append(face)

            X = np.asarray(X).reshape(-1, 48, 48, 1)
            y = to_categorical(emotions, num_classes=7)

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}. Creating sample data...")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration when real dataset is unavailable"""
        logger.info("Creating sample training data...")
        # Create 1000 random samples
        n_samples = 1000
        X = np.random.rand(n_samples, 48, 48, 1)
        y = to_categorical(np.random.randint(0, 7, n_samples), num_classes=7)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val

    def build_model(self):
        """Build CNN model"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        """Train the model and save it"""
        logger.info("Loading and preprocessing data...")
        X_train, X_val, y_train, y_val = self.load_and_preprocess_data()

        logger.info("Building model...")
        model = self.build_model()

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )

        logger.info("Training model...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=64),
            validation_data=(X_val, y_val),
            epochs=25,  # Reduced from 50 for faster training
            verbose=1
        )

        # Save the model
        model.save(MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")

        # Plot training history
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['accuracy'], label='accuracy')
            plt.plot(history.history['val_accuracy'], label='val_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig('training_history.png')
            logger.info("Training history saved to training_history.png")
        except Exception as e:
            logger.warning(f"Could not save training plot: {e}")

        return model

    def load_model(self):
        """Load the trained model"""
        if self._loaded:
            return True

        if not os.path.exists(MODEL_PATH):
            logger.info("Model not found, training new model...")
            self.model = self.train_model()
        else:
            logger.info(f"Loading model from {MODEL_PATH}")
            self.model = load_model(MODEL_PATH)

        self._loaded = True
        return True

    def preprocess_image(self, image_input):
        """
        Preprocess image for model prediction

        Args:
            image_input: Can be PIL Image, numpy array, or file path

        Returns:
            np.array: Preprocessed image array
        """
        image = None
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                image = Image.fromarray(image_input)
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = image_input
            else:
                raise ValueError("Unsupported image input type")

            # Convert to grayscale
            image = image.convert('L')

            # Resize to 48x48
            image = image.resize((48, 48))

            # Convert to array and normalize
            image_array = np.array(image) / 255.0
            image_array = image_array.reshape(1, 48, 48, 1)

            # Close image if we opened it from file
            if isinstance(image_input, str):
                image.close()

            return image_array

        except Exception as e:
            if image and isinstance(image_input, str):
                image.close()
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def predict_emotion(self, image_input, source="unknown", user_name=None):
        """
        Predict emotion from image

        Args:
            image_input: Image input (file path, PIL Image, or numpy array)
            source (str): Source of the prediction
            user_name (str): Name of the user

        Returns:
            dict: Contains 'emotion', 'confidence', and 'all_scores'
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")

            # Preprocess image
            image_array = self.preprocess_image(image_input)

            # Make prediction
            predictions = self.model.predict(image_array)[0]

            # Get prediction results
            predicted_class_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_class_idx])

            # Map to emotion label
            predicted_emotion = self.emotion_labels[predicted_class_idx]

            # Create all scores dictionary - convert numpy float32 to Python float
            all_scores = {}
            for i, emotion in enumerate(self.emotion_labels):
                all_scores[emotion] = float(predictions[i])

            # Log prediction to database
            self._log_prediction(
                predicted_emotion,
                confidence,
                image_input if isinstance(image_input, str) else "uploaded_image",
                source,
                user_name
            )

            result = {
                "emotion": predicted_emotion,
                "confidence": float(confidence),  # Ensure it's a Python float
                "all_scores": all_scores,
            }

            logger.info(f"Prediction: {predicted_emotion} ({confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def _log_prediction(self, emotion, confidence, image_path, source, user_name):
        """Log prediction to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO predictions (timestamp, user_name, predicted_emotion, confidence, image_path, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (datetime.now().isoformat(), user_name, emotion, confidence, image_path, source),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")


# Global model instance
_emotion_detector = None


def get_emotion_detector():
    """
    Get global emotion detector instance (singleton pattern)
    """
    global _emotion_detector
    if _emotion_detector is None:
        _emotion_detector = EmotionDetector()
    return _emotion_detector


def predict_emotion(image_input, source="unknown", user_name=None):
    """
    Predict emotion from image using the global detector

    Args:
        image_input: Image input (file path, PIL Image, or numpy array)
        source (str): Source of the prediction
        user_name (str): Name of the user

    Returns:
        dict: Prediction results
    """
    detector = get_emotion_detector()
    if not detector._loaded:
        detector.load_model()
    return detector.predict_emotion(image_input, source, user_name)


if __name__ == "__main__":
    # Train the model if not exists
    try:
        print("🧪 Training Emotion Detection Model...")
        detector = get_emotion_detector()
        detector.load_model()
        print("✅ Model ready!")

        # Test with a sample image
        test_image = Image.new("L", (48, 48), color=128)  # Grayscale test image
        result = predict_emotion(test_image, source="test", user_name="Test User")

        print(f"Test prediction: {result['emotion']} ({result['confidence']:.3f})")
        print("✅ Test completed!")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
