from flask import Flask, render_template, request, jsonify, session, redirect
import cv2
import numpy as np
import mediapipe as mp
import joblib
import base64
import os
import json
from feature_extractor import extract_features

app = Flask(__name__)
app.secret_key = "secret123"   # required for login sessions

selected_glasses = None

# Load ML models
model = joblib.load("face_shape_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# -----------------------------------
# User JSON functions
# -----------------------------------

def load_users():
    if not os.path.exists("users.json"):
        return []
    with open("users.json", "r") as f:
        return json.load(f)

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)

# -----------------------------------
# Pages
# -----------------------------------

@app.route("/")
def home():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        users = load_users()

        for u in users:
            if u["username"] == username and u["password"] == password:
                session["user"] = username
                return redirect("/")

        return "Invalid login"

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        users = load_users()

        for u in users:
            if u["username"] == username:
                return "User already exists"

        users.append({
            "username": username,
            "password": password
        })

        save_users(users)

        return redirect("/login")

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")


# -----------------------------------
# Glasses Selection
# -----------------------------------

@app.route("/select_glasses")
def select_glasses():

    if "user" not in session:
        return redirect("/login")

    return render_template("select_glasses.html")


@app.route("/set_glasses", methods=["POST"])
def set_glasses():
    global selected_glasses
    selected_glasses = request.json["glasses"]
    return jsonify({"status": "ok"})


# -----------------------------------
# Webcam Page
# -----------------------------------

@app.route("/webcam")
def webcam_page():

    if "user" not in session:
        return redirect("/login")

    return render_template("webcam.html")


# -----------------------------------
# Spectacle Overlay
# -----------------------------------

def apply_overlay(image, pts):

    global selected_glasses

    if selected_glasses:
        spectacle_file = f"frames/{selected_glasses}.png"
    else:
        spectacle_file = "frames/aviator.png"

    if not os.path.exists(spectacle_file):
        return image

    spectacle = cv2.imread(spectacle_file, cv2.IMREAD_UNCHANGED)

    left_temple = pts['left_temple']
    right_temple = pts['right_temple']
    nose_bridge = pts['nose_bridge']

    temple_distance = np.linalg.norm(
        np.array(left_temple) - np.array(right_temple)
    )

    width = int(temple_distance * 1.3)
    height = int(width * spectacle.shape[0] / spectacle.shape[1])

    spectacle = cv2.resize(spectacle, (width, height))

    x = left_temple[0] - int(width * 0.1)
    y = nose_bridge[1] - int(height * 0.4)

    # Ensure within bounds
    if y + height > image.shape[0] or x + width > image.shape[1]:
        return image

    alpha = spectacle[:, :, 3] / 255.0

    for c in range(3):
        image[y:y+height, x:x+width, c] = (
            image[y:y+height, x:x+width, c] * (1 - alpha)
            + spectacle[:, :, c] * alpha
        )

    return image


# -----------------------------------
# Prediction API (Webcam Frames)
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["image"]

    encoded = data.split(",")[1]
    img_bytes = base64.b64decode(encoded)

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    features, pts, roll = extract_features(frame, face_mesh)

    if features is None:
        return jsonify({"status": "no_face"})

    features_scaled = scaler.transform([features])

    pred_idx = model.predict(features_scaled)[0]
    face_shape = label_encoder.inverse_transform([pred_idx])[0]

    confidence = model.predict_proba(features_scaled)[0][pred_idx]

    frame = apply_overlay(frame, pts)

    _, buffer = cv2.imencode(".jpg", frame)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "status": "ok",
        "image": jpg_as_text,
        "face_shape": face_shape,
        "confidence": float(confidence)
    })


# -----------------------------------

if __name__ == "__main__":
    app.run(debug=True)