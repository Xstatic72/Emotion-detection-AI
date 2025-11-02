"""
Simple Emotion Detection Model Module
====================================
Simplified emotion detection using a single Hugging Face model for image upload only.
"""

import os
import torch
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
import sqlite3
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionDetector:
    """
    Simple emotion detection class using a single Hugging Face model
    """

    def __init__(self, model_name="dima806/facial_emotions_image_detection"):
        """
        Initialize the emotion detector with the specified model

        Args:
            model_name (str): Hugging Face model identifier
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.emotion_labels = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
        ]
        self.db_path = "emotion_predictions.db"

        # Initialize database
        self._init_database()

        # Load model
        self.load_model()

    def _init_database(self):
        """Initialize SQLite database for storing predictions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create predictions table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    predicted_emotion TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    image_path TEXT,
                    source TEXT DEFAULT 'unknown'
                )
            """)

            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")

    def load_model(self):
        """
        Load the Hugging Face model and processor
        """
        try:
            logger.info(f"Loading emotion detection model: {self.model_name}")

            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name, torch_dtype=torch.float32
            )

            # Set model to evaluation mode
            self.model.eval()

            logger.info("Model loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def preprocess_image(self, image_input):
        """
        Preprocess image for model prediction

        Args:
            image_input: Can be PIL Image, numpy array, or file path

        Returns:
            PIL.Image: Processed image ready for model
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    # BGR to RGB conversion
                    image_input = image_input[:, :, ::-1]
                image = Image.fromarray(image_input)
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = image_input
            else:
                raise ValueError("Unsupported image input type")

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if too large (optimization)
            max_size = 512
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            return image

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def predict_emotion(self, image_input, source="unknown"):
        """
        Predict emotion from image

        Args:
            image_input: Image input (file path, PIL Image, or numpy array)
            source (str): Source of the prediction ('flask', 'upload', etc.)

        Returns:
            dict: Contains 'emotion', 'confidence', and 'all_scores'
        """
        try:
            if self.model is None or self.processor is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")

            # Preprocess image
            image = self.preprocess_image(image_input)

            # Process image for model input
            inputs = self.processor(images=image, return_tensors="pt")

            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get prediction results
            predicted_class_idx = predictions.argmax().item()
            confidence = predictions.max().item()

            # Map to emotion label
            predicted_emotion = self.emotion_labels[predicted_class_idx]

            # Create all scores dictionary
            all_scores = {}
            for i, emotion in enumerate(self.emotion_labels):
                all_scores[emotion] = float(predictions[0][i].item())

            # Log prediction to database
            self._log_prediction(
                predicted_emotion,
                confidence,
                image_input if isinstance(image_input, str) else "uploaded_image",
                source,
            )

            result = {
                "emotion": predicted_emotion,
                "confidence": confidence,
                "all_scores": all_scores,
            }

            logger.info(f"Prediction: {predicted_emotion} ({confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def _log_prediction(self, emotion, confidence, image_path, source):
        """Log prediction to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO predictions (timestamp, predicted_emotion, confidence, image_path, source)
                VALUES (?, ?, ?, ?, ?)
            """,
                (datetime.now().isoformat(), emotion, confidence, image_path, source),
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


def predict_emotion(image_input, source="unknown"):
    """
    Predict emotion from image using the global detector

    Args:
        image_input: Image input (file path, PIL Image, or numpy array)
        source (str): Source of the prediction

    Returns:
        dict: Prediction results
    """
    detector = get_emotion_detector()
    return detector.predict_emotion(image_input, source)


if __name__ == "__main__":
    # Test the model
    try:
        print("🧪 Testing Emotion Detection Model...")
        detector = get_emotion_detector()

        if detector.model is not None:
            print("✅ Model loaded successfully!")

            # Create a test image
            test_image = Image.new("RGB", (224, 224), color="lightblue")
            result = predict_emotion(test_image, source="test")

            print(f"Test prediction: {result['emotion']} ({result['confidence']:.3f})")
            print("✅ Test completed!")
        else:
            print("❌ Model failed to load")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
