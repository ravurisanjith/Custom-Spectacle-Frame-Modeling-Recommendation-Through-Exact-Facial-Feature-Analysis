import numpy as np
import math
import cv2  # Add this as your existing extract_features uses cv2

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_additional_features(pts):
    features = {}
    features['forehead_width'] = np.linalg.norm(np.array(pts['left_temple']) - np.array(pts['right_temple']))
    features['chin_width'] = np.linalg.norm(np.array(pts['left_jaw']) - np.array(pts['right_jaw']))
    features['jaw_angle'] = calculate_angle(pts['left_jaw'], pts['chin'], pts['right_jaw'])
    cheek_width = np.linalg.norm(np.array(pts['left_cheek']) - np.array(pts['right_cheek']))
    jaw_width = features['chin_width']
    features['cheek_jaw_ratio'] = cheek_width / jaw_width if jaw_width != 0 else 0
    features['nose_length'] = np.linalg.norm(np.array(pts['nose_bridge']) - np.array(pts['chin']))
    return features

def extract_features(image, face_mesh):
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, None, None

    lm = results.multi_face_landmarks[0].landmark
    landmark_ids = {
        'left_eye_outer': 33,
        'right_eye_outer': 263,
        'nose_bridge': 168,
        'chin': 152,
        'left_jaw': 234,
        'right_jaw': 454,
        'left_cheek': 127,
        'right_cheek': 356,
        'forehead': 10,
        'left_temple': 21,
        'right_temple': 251
    }
    pts = {name: (int(lm[idx].x * w), int(lm[idx].y * h)) for name, idx in landmark_ids.items()}

    jaw_width = np.linalg.norm(np.array(pts['left_jaw']) - np.array(pts['right_jaw']))
    face_height = np.linalg.norm(np.array(pts['forehead']) - np.array(pts['chin']))
    eye_distance = np.linalg.norm(np.array(pts['left_eye_outer']) - np.array(pts['right_eye_outer']))
    cheek_width = np.linalg.norm(np.array(pts['left_cheek']) - np.array(pts['right_cheek']))
    temple_width = np.linalg.norm(np.array(pts['left_temple']) - np.array(pts['right_temple']))

    face_ratio = face_height / jaw_width if jaw_width else 0
    jaw_to_temple_ratio = jaw_width / temple_width if temple_width else 0
    eye_to_jaw_ratio = eye_distance / jaw_width if jaw_width else 0

    left_eye = lm[landmark_ids['left_eye_outer']]
    right_eye = lm[landmark_ids['right_eye_outer']]
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    roll = math.degrees(math.atan2(dy, dx))

    additional_feats = extract_additional_features(pts)

    all_features = [
        jaw_width,
        face_height,
        eye_distance,
        cheek_width,
        temple_width,
        face_ratio,
        jaw_to_temple_ratio,
        eye_to_jaw_ratio,
        additional_feats['forehead_width'],
        additional_feats['chin_width'],
        additional_feats['jaw_angle'],
        additional_feats['cheek_jaw_ratio'],
        additional_feats['nose_length']
    ]

    return all_features, pts, roll
