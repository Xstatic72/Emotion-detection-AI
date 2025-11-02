from flask import Flask, render_template, request, jsonify
import os
import tempfile
from werkzeug.utils import secure_filename
from PIL import Image
import sqlite3
from datetime import datetime
import logging
from model import get_emotion_detector, predict_emotion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "emotion_detection_secret_key_2024"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"}

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_database_stats():
    try:
        conn = sqlite3.connect("emotion_predictions.db")
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        cursor.execute(
            "SELECT predicted_emotion, COUNT(*) FROM predictions GROUP BY predicted_emotion ORDER BY COUNT(*) DESC"
        )
        emotion_counts = dict(cursor.fetchall())

        cursor.execute(
            "SELECT timestamp, predicted_emotion, confidence, source FROM predictions ORDER BY timestamp DESC LIMIT 5"
        )
        recent_predictions = cursor.fetchall()

        conn.close()

        return {
            "total_predictions": total_predictions,
            "emotion_counts": emotion_counts,
            "recent_predictions": recent_predictions,
        }
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        return {"total_predictions": 0, "emotion_counts": {}, "recent_predictions": []}


@app.route("/")
def index():
    try:
        stats = get_database_stats()
        return render_template(
            "index.html", stats=stats, title="Emotion Detection Web App"
        )
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return render_template(
            "index.html",
            stats={
                "total_predictions": 0,
                "emotion_counts": {},
                "recent_predictions": [],
            },
            title="Emotion Detection Web App",
        )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify(
                {"error": "File type not allowed. Please upload an image file."}
            ), 400

        if file and allowed_file(file.filename):
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".jpg"
                ) as temp_file:
                    file.save(temp_file.name)

                    try:
                        img = Image.open(temp_file.name)
                        img.verify()
                        img = Image.open(temp_file.name)

                        if img.mode != "RGB":
                            img = img.convert("RGB")

                    except Exception as img_error:
                        os.unlink(temp_file.name)
                        return jsonify(
                            {"error": f"Invalid image file: {str(img_error)}"}
                        ), 400

                    logger.info(
                        f"Making prediction for uploaded image: {file.filename}"
                    )
                    result = predict_emotion(temp_file.name, source="flask")

                    os.unlink(temp_file.name)

                    response_data = {
                        "success": True,
                        "predicted_emotion": result["emotion"],
                        "confidence": round(result["confidence"], 3),
                        "all_scores": {
                            k: round(v, 3) for k, v in result["all_scores"].items()
                        },
                        "filename": secure_filename(file.filename),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    logger.info(
                        f"Prediction successful: {result['emotion']} ({result['confidence']:.3f})"
                    )
                    return jsonify(response_data)

            except Exception as process_error:
                logger.error(f"Error processing image: {str(process_error)}")
                return jsonify(
                    {"error": f"Error processing image: {str(process_error)}"}
                ), 500

    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/health")
def health_check():
    try:
        conn = sqlite3.connect("emotion_predictions.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        conn.close()

        detector = get_emotion_detector()
        model_loaded = detector.model is not None

        return jsonify(
            {
                "status": "healthy",
                "model_loaded": model_loaded,
                "database_predictions": count,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify(
            {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
        ), 500


@app.route("/stats")
def stats():
    try:
        stats = get_database_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413


if __name__ == "__main__":
    try:
        logger.info("Starting Flask application...")
        detector = get_emotion_detector()
        logger.info("Model initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")

    # Use environment variables for deployment
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"

    app.run(host="0.0.0.0", port=port, debug=debug)
