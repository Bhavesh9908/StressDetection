import os
import uuid
import base64
import cv2
import numpy as np
import logging
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import cloudinary
import cloudinary.uploader
from PIL import Image, ImageDraw, ImageFont

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Configure Cloudinary
cloudinary.config(
    cloud_name="da4mdjezu",
    api_key="493281977135412",
    api_secret="P5xxU64uEjNZy6wITFM5pD5Qu54"
)

# Load model
logging.debug("Loading model...")
model = load_model('stressdetection.hdf5', compile=False)
logging.debug("Model loaded successfully.")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
stress_emotions = ['Angry', 'Disgust', 'Fear', 'Sad']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def allowed_file(file):
    return file and file.content_length <= MAX_FILE_SIZE

def compress_image_to_target_size(image_path, target_size_kb=30):
    """Compress image to be under target size in KB."""
    target_bytes = target_size_kb * 1024
    img = Image.open(image_path).convert("RGB")

    compressed_path = image_path.replace(".png", ".jpg").replace(".jpeg", ".jpg")
    quality = 95
    last_good_path = None

    for q in range(quality, 10, -5):
        img.save(compressed_path, format="JPEG", quality=q)
        current_size = os.path.getsize(compressed_path)

        if current_size <= target_bytes:
            return compressed_path  # Return compressed if successful

        last_good_path = compressed_path

    logging.warning("Could not compress below 30KB. Using best-effort version.")
    return last_good_path or image_path

@app.route("/", methods=["GET", "POST"])
def index():
    logging.debug("Inside index route")
    result = None
    image_path = None

    if request.method == "POST":
        if "image" in request.files:
            file = request.files["image"]
            if not allowed_file(file):
                return jsonify({"error": "File too large. Please upload a smaller image."}), 400

            if file:
                filename = secure_filename(file.filename)
                local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                try:
                    file.save(local_path)
                    logging.debug(f"File saved: {local_path}")
                except Exception as e:
                    logging.error(f"Error saving file: {e}")
                    return jsonify({"error": "Error saving file"}), 500

                # Compress to 30KB
                local_path = compress_image_to_target_size(local_path)

                # Upload to Cloudinary
                try:
                    upload_result = cloudinary.uploader.upload(local_path)
                    image_url = upload_result['secure_url']
                except Exception as e:
                    logging.error(f"Cloudinary upload failed: {e}")
                    return jsonify({"error": "Cloudinary upload failed"}), 500

                # Predict emotion
                try:
                    result = predict_emotion(local_path)
                except Exception as e:
                    logging.error(f"Prediction failed: {e}")
                    return jsonify({"error": "Prediction failed"}), 500

                image_path = image_url

    return render_template("index.html", result=result, image_path=image_path)

@app.route("/capture", methods=["POST"])
def capture():
    data = request.form["image_data"].split(",")[1]
    img_bytes = base64.b64decode(data)
    filename = f"{uuid.uuid4().hex}.png"
    local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    with open(local_path, "wb") as f:
        f.write(img_bytes)

    # Compress to 30KB
    local_path = compress_image_to_target_size(local_path)

    # Upload to Cloudinary
    try:
        upload_result = cloudinary.uploader.upload(local_path)
        image_url = upload_result['secure_url']
    except Exception as e:
        logging.error(f"Cloudinary upload failed: {e}")
        return jsonify({"error": "Cloudinary upload failed"}), 500

    result = predict_emotion(local_path)
    return render_template("index.html", result=result, image_path=image_url)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not allowed_file(file):
        return jsonify({"error": "File too large. Please upload a smaller image."}), 400

    if file:
        filename = secure_filename(file.filename)
        local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(local_path)

        # Compress to 30KB
        local_path = compress_image_to_target_size(local_path)

        result = predict_emotion(local_path)
        return jsonify(result)

    return jsonify({"error": "Invalid image upload"}), 400

def annotate_image_with_info(image_path, emotion, stress, stress_score):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 25)
    except:
        font = ImageFont.load_default()

    text_lines = [
        f"Emotion: {emotion}",
        f"Stress: {stress}",
        f"Stress Score: {stress_score}"
    ]

    x, y = 10, 10
    for line in text_lines:
        draw.text((x, y), line, fill="red", font=font)
        y += 30

    annotated_path = "static/annotated_result.jpg"
    img.save(annotated_path)

    return annotated_path

def predict_emotion(image_path):
    logging.debug(f"Predicting emotion for image: {image_path}")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        logging.warning("No face detected.")
        return {"error": "No face detected."}

    (x, y, w, h) = faces[0]
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (64, 64)).astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)[..., np.newaxis]

    preds = model.predict(roi, verbose=0)[0]
    emotion_idx = np.argmax(preds)
    emotion = emotion_labels[emotion_idx]
    confidence = preds[emotion_idx] * 100
    stress_score = sum([preds[emotion_labels.index(e)] for e in stress_emotions]) * 100
    stress = "Stress" if stress_score >= 50 else "Non-Stress"

    annotated_image_path = annotate_image_with_info(
        image_path, emotion, stress, f"{stress_score:.1f}%"
    )

    cloudinary_result = cloudinary.uploader.upload(annotated_image_path)
    annotated_image_url = cloudinary_result['secure_url']

    return {
        "emotion": emotion,
        "confidence": f"{confidence:.1f}%",
        "stress": stress,
        "stress_score": f"{stress_score:.1f}%",
        "image_url": annotated_image_url
    }

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
