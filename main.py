import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import numpy as np
import joblib
import os
from feature_extractor import extract_features

# Load model, scaler and label encoder
try:
    model = joblib.load('face_shape_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except:
    messagebox.showerror("Error", "Could not load model files. Make sure face_shape_model.pkl, scaler.pkl, and label_encoder.pkl exist.")
    exit()

mp_face_mesh = mp.solutions.face_mesh

# Face shape to spectacle mapping
face_shape_to_spectacle = {
    'oval': 'frames/aviator.png',
    'round': 'frames/round.png', 
    'square': 'frames/rectangular.png',
    'heart': 'frames/cateye.png',
    'oblong': 'frames/large.png'
}

class FaceShapeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Shape & Spectacle Recommender")
        self.root.geometry("800x600")
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.cap = None
        self.running = False
        self.last_measurements = None
        self.last_face_shape = None
        self.last_measurements_cm = None
        self.last_frame_width_cm = None
        
        self.setup_initial_screen()
        
    def setup_initial_screen(self):
        self.clear_screen()
        
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=True)
        
        title = tk.Label(main_frame, text="Face Shape & Spectacle Recommender", 
                        font=("Arial", 18, "bold"))
        title.pack(pady=20)
        
        tk.Label(main_frame, text="Choose Input Mode:", 
                font=("Arial", 14)).pack(pady=10)
        
        btn_photo = tk.Button(main_frame, text="Upload Photo", 
                             font=("Arial", 12), width=20, height=2,
                             command=self.setup_photo_mode)
        btn_photo.pack(pady=10)
        
        btn_webcam = tk.Button(main_frame, text="Use Webcam", 
                              font=("Arial", 12), width=20, height=2,
                              command=self.setup_webcam_mode)
        btn_webcam.pack(pady=10)
        
    def setup_photo_mode(self):
        self.clear_screen()
        
        # Top buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Load Image", 
                 command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Show Details", 
                 command=self.show_details).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Back", 
                 command=self.setup_initial_screen).pack(side=tk.LEFT, padx=5)
        
        # Canvas for image
        self.photo_canvas = tk.Canvas(self.root, width=640, height=480, bg='black')
        self.photo_canvas.pack(pady=10)
        
        # Result label
        self.photo_result = tk.Label(self.root, text="Load an image to begin", 
                                   font=("Arial", 12))
        self.photo_result.pack()
        
    def setup_webcam_mode(self):
        self.clear_screen()
        
        # Top buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.btn_start = tk.Button(button_frame, text="Start Webcam", 
                                  command=self.start_webcam)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = tk.Button(button_frame, text="Stop Webcam", 
                                 command=self.stop_webcam, state='disabled')
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Show Details", 
                 command=self.show_details).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Back", 
                 command=self.setup_initial_screen).pack(side=tk.LEFT, padx=5)
        
        # Canvas for webcam
        self.webcam_canvas = tk.Canvas(self.root, width=640, height=480, bg='black')
        self.webcam_canvas.pack(pady=10)
        
        # Result label
        self.webcam_result = tk.Label(self.root, text="Click Start Webcam to begin", 
                                    font=("Arial", 12))
        self.webcam_result.pack()
        
        # Frame size label
        self.frame_size_label = tk.Label(self.root, text="", 
                                       font=("Arial", 11, "bold"), fg="blue")
        self.frame_size_label.pack()
        
    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()
            
    def pixels_to_cm(self, pixels, face_width_pixels):
        # Average human face width is about 14.5 cm
        # Use this to calibrate pixel to cm conversion
        avg_face_width_cm = 14.5
        pixels_per_cm = face_width_pixels / avg_face_width_cm
        return pixels / pixels_per_cm
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
            
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Could not load image")
            return
            
        # Process image
        result = self.analyze_face(image)
        if result:
            pred_label, confidence, measurements, overlay_image, measurements_cm, frame_width_cm = result
            self.last_face_shape = pred_label
            self.last_measurements = measurements
            self.last_measurements_cm = measurements_cm
            self.last_frame_width_cm = frame_width_cm
            
            # Show processed image
            self.display_image(overlay_image, self.photo_canvas)
            self.photo_result.config(text=f"Face Shape: {pred_label} (Confidence: {confidence:.2f})\nRecommended Frame Width: {frame_width_cm:.1f} cm")
        else:
            self.photo_result.config(text="No face detected in image")
            
    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not access webcam")
            return
            
        self.running = True
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.update_webcam()
        
    def stop_webcam(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.webcam_canvas.delete("all")
        self.webcam_result.config(text="Webcam stopped")
        self.frame_size_label.config(text="")
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        
    def update_webcam(self):
        if not self.running or not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop_webcam()
            return
            
        frame = cv2.flip(frame, 1)  # Mirror effect
        
        # Process frame
        result = self.analyze_face(frame, dynamic=True)
        if result:
            pred_label, confidence, measurements, overlay_image, measurements_cm, frame_width_cm = result
            self.last_face_shape = pred_label
            self.last_measurements = measurements
            self.last_measurements_cm = measurements_cm
            self.last_frame_width_cm = frame_width_cm
            
            self.display_image(overlay_image, self.webcam_canvas)
            self.webcam_result.config(text=f"Face Shape: {pred_label} (Confidence: {confidence:.2f})")
            self.frame_size_label.config(text=f"Recommended Frame Width: {frame_width_cm:.1f} cm")
        else:
            # Just show the frame without overlay
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_image(frame_rgb, self.webcam_canvas)
            self.webcam_result.config(text="No face detected")
            self.frame_size_label.config(text="")
            
        self.root.after(30, self.update_webcam)
        
    def analyze_face(self, image, dynamic=False):
        try:
            # Extract features using your feature extractor
            features, pts, roll = extract_features(image, self.face_mesh)
            
            if features is None or pts is None:
                return None
                
            # Predict face shape
            features_scaled = scaler.transform([features])
            pred_idx = model.predict(features_scaled)[0]
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            confidence = model.predict_proba(features_scaled)[0][pred_idx]
            
            # Prepare measurements in pixels
            measurements = {
                "Jaw Width": features[0],
                "Face Height": features[1], 
                "Eye Distance": features[2],
                "Cheek Width": features[3],
                "Temple Width": features[4],
                "Face Ratio": features[5],
                "Jaw to Temple Ratio": features[6],
                "Eye to Jaw Ratio": features[7]
            }
            
            # Convert measurements to cm using face width as reference
            face_width_pixels = features[4]  # Temple width as reference
            measurements_cm = {}
            for key, value in measurements.items():
                if key not in ["Face Ratio", "Jaw to Temple Ratio", "Eye to Jaw Ratio"]:
                    measurements_cm[key] = self.pixels_to_cm(value, face_width_pixels)
                else:
                    measurements_cm[key] = value  # Ratios remain unitless
            
            # Calculate recommended frame width (slightly smaller than temple width)
            frame_width_cm = measurements_cm["Temple Width"] * 0.85
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply spectacle overlay
            overlay_image = self.apply_spectacle_overlay(image_rgb, pred_label, pts, 
                                                       features, roll, dynamic)
            
            return pred_label, confidence, measurements, overlay_image, measurements_cm, frame_width_cm
            
        except Exception as e:
            print(f"Error in analyze_face: {e}")
            return None
            
    def apply_spectacle_overlay(self, image, face_shape, pts, features, roll, dynamic=False):
        try:
            # Get spectacle file for this face shape
            spectacle_file = face_shape_to_spectacle.get(face_shape, 'frames/aviator.png')
            
            if not os.path.exists(spectacle_file):
                print(f"Spectacle file not found: {spectacle_file}")
                return image
                
            spectacle = cv2.imread(spectacle_file, cv2.IMREAD_UNCHANGED)
            if spectacle is None:
                print(f"Could not load spectacle: {spectacle_file}")
                return image
                
            # Get key points
            left_temple = pts['left_temple']
            right_temple = pts['right_temple'] 
            nose_bridge = pts['nose_bridge']
            
            # Calculate spectacle size - make it larger and more realistic
            temple_distance = np.linalg.norm(np.array(left_temple) - np.array(right_temple))
            
            # Make spectacles larger - use temple width + some extra
            width = int(temple_distance * 1.3)  # Increased from 1.0 to 1.3
            
            # Dynamic scaling for webcam
            if dynamic:
                face_height = features[1]
                # Better scaling based on face size in frame
                if face_height < 200:  # Small face (far from camera)
                    scale_factor = 0.8
                elif face_height > 400:  # Large face (close to camera)
                    scale_factor = 1.2
                else:
                    scale_factor = 1.0
                width = int(width * scale_factor)
            
            height = int(width * spectacle.shape[0] / spectacle.shape[1])
            
            # Ensure reasonable size
            width = max(width, 100)  # Increased minimum size
            height = max(height, 40)  # Increased minimum size
            
            # Resize spectacle
            spectacle_resized = cv2.resize(spectacle, (width, height))
            
            # Apply rotation for head tilt
            if abs(roll) > 2:
                center = (width // 2, height // 2)
                M = cv2.getRotationMatrix2D(center, -roll, 1.0)
                spectacle_resized = cv2.warpAffine(spectacle_resized, M, (width, height),
                                                 borderMode=cv2.BORDER_CONSTANT, 
                                                 borderValue=(0, 0, 0, 0))
            
            # Position spectacle - better positioning
            x = left_temple[0] - int(width * 0.1)  # Slightly to the left
            y = nose_bridge[1] - int(height * 0.4)  # Higher up
            
            # Ensure spectacle fits in image
            x = max(0, min(x, image.shape[1] - width))
            y = max(0, min(y, image.shape[0] - height))
            
            # Apply overlay with alpha blending
            if spectacle_resized.shape[2] == 4:  # Has alpha channel
                overlay_region = image[y:y+height, x:x+width]
                alpha = spectacle_resized[:, :, 3:4] / 255.0
                
                # Blend the spectacle onto the image
                blended = overlay_region * (1 - alpha) + spectacle_resized[:, :, :3] * alpha
                image[y:y+height, x:x+width] = blended.astype(np.uint8)
            else:
                # Simple overlay without alpha
                image[y:y+height, x:x+width] = spectacle_resized[:, :, :3]
                
            return image
            
        except Exception as e:
            print(f"Error applying overlay: {e}")
            return image
            
    def display_image(self, image, canvas):
        try:
            # Resize image to fit canvas
            height, width = image.shape[:2]
            canvas_width = 640
            canvas_height = 480
            # Calculate scaling to fit canvas
            scale = min(canvas_width/width, canvas_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            image_resized = cv2.resize(image, (new_width, new_height))
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(image_resized)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            canvas.delete("all")
            canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
            canvas.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            
    def show_details(self):
        if not self.last_measurements_cm or not self.last_face_shape:
            messagebox.showinfo("No Data", "Please analyze a face first")
            return
            
        # Create details window
        details_window = tk.Toplevel(self.root)
        details_window.title("Face Shape Details")
        details_window.geometry("450x600")
        
        # Main frame with scrollbar
        main_frame = tk.Frame(details_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Face shape
        tk.Label(main_frame, text=f"Face Shape: {self.last_face_shape}", 
                font=("Arial", 16, "bold"), fg="darkblue").pack(pady=10)
        
        # Recommended frame width
        tk.Label(main_frame, text=f"Recommended Frame Width: {self.last_frame_width_cm:.1f} cm", 
                font=("Arial", 14, "bold"), fg="darkgreen", bg="lightgreen").pack(pady=10, fill='x')
        
        # Measurements in CM
        tk.Label(main_frame, text="Facial Measurements (in centimeters):", 
                font=("Arial", 12, "bold")).pack(pady=(20,10))
        
        measurements_frame = tk.Frame(main_frame, relief='groove', bd=2)
        measurements_frame.pack(fill='x', pady=5)
        
        for key, value in self.last_measurements_cm.items():
            frame = tk.Frame(measurements_frame)
            frame.pack(fill='x', padx=10, pady=2)
            
            if key not in ["Face Ratio", "Jaw to Temple Ratio", "Eye to Jaw Ratio"]:
                tk.Label(frame, text=f"• {key}:", font=("Arial", 10), anchor='w').pack(side='left')
                tk.Label(frame, text=f"{value:.1f} cm", font=("Arial", 10, "bold"), 
                        fg="blue", anchor='e').pack(side='right')
            else:
                tk.Label(frame, text=f"• {key}:", font=("Arial", 10), anchor='w').pack(side='left')
                tk.Label(frame, text=f"{value:.2f}", font=("Arial", 10, "bold"), 
                        fg="blue", anchor='e').pack(side='right')
        
        # Spectacle sizing guide
        tk.Label(main_frame, text="Spectacle Sizing Guide:", 
                font=("Arial", 12, "bold")).pack(pady=(20,10))
        
        guide_frame = tk.Frame(main_frame, relief='groove', bd=2, bg='lightyellow')
        guide_frame.pack(fill='x', pady=5)
        
        guide_text = f"""
• Your face shape: {self.last_face_shape.upper()}
• Temple width: {self.last_measurements_cm['Temple Width']:.1f} cm
• Eye distance: {self.last_measurements_cm['Eye Distance']:.1f} cm
• Recommended frame width: {self.last_frame_width_cm:.1f} cm
• Frame should not exceed temple width
• Ideal frame covers about 85% of face width
        """
        
        tk.Label(guide_frame, text=guide_text.strip(), font=("Arial", 9), 
                justify='left', bg='lightyellow').pack(padx=10, pady=10)
        
        # Close button
        tk.Button(main_frame, text="Close", font=("Arial", 12),
                 command=details_window.destroy).pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceShapeApp(root)
    
    def on_closing():
        if app.running:
            app.stop_webcam()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


