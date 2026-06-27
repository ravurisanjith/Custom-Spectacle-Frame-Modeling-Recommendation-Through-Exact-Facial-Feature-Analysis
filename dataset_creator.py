# Face Shape Dataset Creator and Feature Extractor
# This script helps you organize images and extract features for face shape classification

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Create dataset folder structure
def create_dataset_structure():
    """Creates the required folder structure for face shape dataset"""
    shapes = ['oval', 'round', 'square', 'heart', 'oblong']
    
    # Create main dataset folder
    Path('face_shape_dataset').mkdir(exist_ok=True)
    
    # Create subfolders for each shape
    for shape in shapes:
        Path(f'face_shape_dataset/{shape}').mkdir(exist_ok=True)
    
    print("Dataset folder structure created!")
    print("Please add 10-15 images to each folder:")
    for shape in shapes:
        print(f"  - face_shape_dataset/{shape}/")

# Feature extraction using MediaPipe
class FaceFeatureExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Key landmark indices
        self.landmarks = {
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
    
    def extract_features(self, image_path):
        """Extract facial features from an image"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        h, w = img.shape[:2]
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Extract coordinates
        pts = {}
        for name, idx in self.landmarks.items():
            pts[name] = (
                int(landmarks[idx].x * w),
                int(landmarks[idx].y * h)
            )
        
        # Calculate measurements
        features = self.calculate_measurements(pts)
        return features
    
    def calculate_measurements(self, pts):
        """Calculate facial measurements from landmark points"""
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Key measurements
        jaw_width = distance(pts['left_jaw'], pts['right_jaw'])
        face_height = distance(pts['forehead'], pts['chin'])
        eye_distance = distance(pts['left_eye_outer'], pts['right_eye_outer'])
        cheek_width = distance(pts['left_cheek'], pts['right_cheek'])
        temple_width = distance(pts['left_temple'], pts['right_temple'])
        
        # Calculate ratios (important for face shape classification)
        face_ratio = face_height / jaw_width if jaw_width > 0 else 0
        jaw_to_temple_ratio = jaw_width / temple_width if temple_width > 0 else 0
        eye_to_jaw_ratio = eye_distance / jaw_width if jaw_width > 0 else 0
        
        return {
            'jaw_width': jaw_width,
            'face_height': face_height,
            'eye_distance': eye_distance,
            'cheek_width': cheek_width,
            'temple_width': temple_width,
            'face_ratio': face_ratio,
            'jaw_to_temple_ratio': jaw_to_temple_ratio,
            'eye_to_jaw_ratio': eye_to_jaw_ratio
        }

def process_dataset():
    """Process all images in the dataset and extract features"""
    extractor = FaceFeatureExtractor()
    data = []
    
    shapes = ['oval', 'round', 'square', 'heart', 'oblong']
    
    for shape in shapes:
        folder_path = f'face_shape_dataset/{shape}'
        if not os.path.exists(folder_path):
            continue
            
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                features = extractor.extract_features(image_path)
                
                if features:
                    features['filename'] = filename
                    features['face_shape'] = shape
                    features['image_path'] = image_path
                    data.append(features)
                    print(f"Processed: {filename} -> {shape}")
                else:
                    print(f"Failed to process: {filename}")
    
    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv('face_shape_features.csv', index=False)
        print(f"\nFeature extraction complete!")
        print(f"Total images processed: {len(data)}")
        print(f"Features saved to: face_shape_features.csv")
        
        # Display summary
        print("\nDataset summary:")
        print(df['face_shape'].value_counts())
    else:
        print("No images found or processed successfully!")

def create_labels_csv():
    """Create a simple labels CSV file"""
    data = []
    shapes = ['oval', 'round', 'square', 'heart', 'oblong']
    
    for shape in shapes:
        folder_path = f'face_shape_dataset/{shape}'
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    data.append({
                        'filename': filename,
                        'face_shape': shape,
                        'folder': shape
                    })
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv('labels.csv', index=False)
        print(f"Labels CSV created with {len(data)} entries")

if __name__ == "__main__":
    print("Face Shape Dataset Creator")
    print("=" * 30)
    
    # Step 1: Create folder structure
    create_dataset_structure()
    
    print("\nNext steps:")
    print("1. Add 10-15 clear frontal face images to each shape folder")
    print("2. Run this script again to extract features")
    print("3. Use 'face_shape_features.csv' for training your model")
    
    # Check if images exist and process them
    input("\nPress Enter after adding images to continue with feature extraction...")
    
    # Step 2: Process images and extract features
    process_dataset()
    
    # Step 3: Create labels CSV
    create_labels_csv()
    
    print("\nDataset creation complete! Files created:")
    print("- face_shape_dataset/ (folder structure)")
    print("- face_shape_features.csv (extracted features)")
    print("- labels.csv (image labels)")