#!/usr/bin/env python3
"""
Flask Emotion Detection Web App Launcher
========================================
A simple script to run the Flask-based emotion detection web application.
"""

import os
import sys
import subprocess
import logging
import socket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if required packages are installed"""
    required_packages = ["flask", "torch", "transformers", "PIL", "numpy"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please install them using: pip install -r requirements.txt")
        return False

    return True


def find_free_port(start_port=5000, max_attempts=10):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    return None


def test_model():
    """Test model loading"""
    try:
        logger.info("Testing model loading...")
        from model import get_emotion_detector

        detector = get_emotion_detector()
        if detector.model is not None:
            logger.info("✅ Model loaded successfully")
            return True
        else:
            logger.warning(
                "⚠️ Model loading issue - check your internet connection for first run"
            )
            return True  # Still allow app to run
    except Exception as e:
        logger.warning(f"⚠️ Model test failed: {e}")
        return True  # Still allow app to run


def main():
    """Main launcher function"""
    print("🎭 Emotion Detection Web App Launcher")
    print("=" * 45)

    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        logger.error(
            "app.py not found. Please run this script from the project directory."
        )
        sys.exit(1)

    # Check requirements
    logger.info("Checking requirements...")
    if not check_requirements():
        sys.exit(1)

    logger.info("✅ All requirements satisfied")

    # Test model loading
    test_model()

    # Find available port
    port = find_free_port()
    if port is None:
        logger.error(
            "Could not find an available port. Please free up ports 5000-5009."
        )
        sys.exit(1)

    # Start Flask app
    logger.info("Starting Flask application...")
    logger.info(f"🌐 App will be available at: http://localhost:{port}")
    logger.info("Press Ctrl+C to stop the server")

    try:
        # Import and run the Flask app
        from app import app

        app.run(host="0.0.0.0", port=port, debug=False)
    except KeyboardInterrupt:
        logger.info("\n👋 Application stopped by user")
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
