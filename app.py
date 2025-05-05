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

# Load the model globally
logging.debug("Loading model...")
model = load_model('stressdetection.hdf5', compile=False)
logging.debug("Model loaded successfully.")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
stress_emotions = ['Angry', 'Disgust', 'Fear', 'Sad']

@app.route("/", methods=["GET", "POST"])
def index():
    logging.debug("Inside index route")
    result = None
    image_path = None

    if request.method == "POST":
        if "image" in request.files:
            file = request.files["image"]
            if file:
                filename = secure_filename(file.filename)
                local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(local_path)

                # Upload to Cloudinary
                upload_result = cloudinary.uploader.upload(local_path)
                image_url = upload_result['secure_url']

                result = predict_emotion(local_path)
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

    # Upload to Cloudinary
    upload_result = cloudinary.uploader.upload(local_path)
    image_url = upload_result['secure_url']

    result = predict_emotion(local_path)
    return render_template("index.html", result=result, image_path=image_url)

# ➡️ New API route that returns JSON for your app
@app.route("/api/predict", methods=["POST"])
def api_predict():
    logging.debug("Inside API predict route")
    
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file:
        filename = secure_filename(file.filename)
        local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(local_path)

        # Predict
        result = predict_emotion(local_path)

        # Return result as JSON
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
    app.run(host="0.0.0.0", port=port, debug=True)