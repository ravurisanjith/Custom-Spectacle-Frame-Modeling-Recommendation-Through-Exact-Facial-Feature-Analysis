import os
import csv
import cv2
from feature_extractor import extract_features
import mediapipe as mp

# Paths
dataset_dir = 'face_shape_dataset'  # Change if your dataset folder name differs
output_csv = 'face_shape_features.csv'

# Face shape classes (folder names)
face_shapes = ['oval', 'round', 'square', 'heart', 'oblong']

mp_face_mesh = mp.solutions.face_mesh

def main():
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        with open(output_csv, mode='w', newline='') as csvfile:
            # Define CSV headers for all features
            header = [
                'jaw_width', 'face_height', 'eye_distance', 'cheek_width', 'temple_width',
                'face_ratio', 'jaw_to_temple_ratio', 'eye_to_jaw_ratio',
                'forehead_width', 'chin_width', 'jaw_angle', 'cheek_jaw_ratio', 'nose_length',
                'face_shape'
            ]
            writer = csv.writer(csvfile)
            writer.writerow(header)

            for face_shape in face_shapes:
                folder_path = os.path.join(dataset_dir, face_shape)
                if not os.path.isdir(folder_path):
                    print(f"Warning: Folder not found {folder_path}")
                    continue
                for filename in os.listdir(folder_path):
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    img_path = os.path.join(folder_path, filename)
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Warning: Could not open image {img_path}")
                        continue

                    features, pts, roll = extract_features(image, face_mesh)
                    if features is None:
                        print(f"Warning: Face not detected in image {img_path}")
                        continue

                    writer.writerow(features + [face_shape])
                    print(f"Processed {img_path}")

    print(f"Feature extraction complete. Saved to {output_csv}")

if __name__ == "__main__":
    main()
