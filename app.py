import os
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS

from predict import predict_audio

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def generate_explanation(features, prediction, confidence):
    explanation = []

    flat_features = features.flatten()
    variance = float(flat_features.var())
    mean_value = float(flat_features.mean())

    if variance < 50:
        explanation.append("Low MFCC variance was detected, which can be associated with synthetic or overly uniform speech patterns.")
    else:
        explanation.append("The extracted MFCC profile shows a more varied acoustic pattern, which is more typical of natural human speech.")

    if confidence >= 0.85:
        explanation.append("The model produced a high-confidence classification for this sample.")
    elif confidence >= 0.65:
        explanation.append("The model produced a moderate-confidence classification, so the result should be interpreted with some caution.")
    else:
        explanation.append("The model confidence is relatively low, indicating a less certain classification.")

    if prediction == 1:
        explanation.append("The sample contains characteristics that align more closely with AI-generated or manipulated speech.")
    else:
        explanation.append("The sample contains fluctuations and feature behaviour more consistent with authentic human speech.")

    explanation.append(f"Feature summary: mean={mean_value:.2f}, variance={variance:.2f}.")

    return explanation


@app.route("/predict", methods=["POST"])
def predict_route():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    file = request.files["audio"]

    if not file or not file.filename:
        return jsonify({"error": "No audio file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported file type. Please upload WAV, MP3, FLAC, M4A, AAC, or OGG."
        }), 400

    suffix = os.path.splitext(file.filename)[1].lower() or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        file_path = temp_file.name

    file.save(file_path)

    try:
        result = predict_audio(file_path)

        prediction = result["prediction"]
        confidence = result["confidence"]
        features = result["features"]
        sample_rate = result["sample_rate"]
        duration = result["duration"]

        explanation = generate_explanation(features, prediction, confidence)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "sample_rate": sample_rate,
            "duration": duration,
            "filename": file.filename,
            "explanation": explanation
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "EchoGuard backend is running."
    })


if __name__ == "__main__":
    app.run(debug=True)