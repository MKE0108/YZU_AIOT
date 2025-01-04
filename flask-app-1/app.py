from flask import Flask, Response, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import torch.nn as nn
import torch
import os
from torch.utils.data import Dataset, DataLoader
import train
import shutil

app = Flask(__name__)
#
last_prediction = None

# Define binary classification model
class PostureClassifier(nn.Module):
    def __init__(self, input_dim):
        super(PostureClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability between 0-1
        )

    def forward(self, x):
        return self.fc(x)

# Load trained model
input_dim = 39  # Number of landmark features (13 points, each with x, y, z)
model = PostureClassifier(input_dim)
#if posture_classifier.pth exists, load it
if os.path.exists("posture_classifier.pth"):
    model.load_state_dict(torch.load("posture_classifier.pth"))
model.eval()

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

recording_normal = False
recording_abnormal = False
normal_data_count = 0
abnormal_data_count = 0

#create folder
os.makedirs('data/normal', exist_ok=True)
os.makedirs('data/abnormal', exist_ok=True)


def generate_frames():
    global normal_data_count, abnormal_data_count, model,last_prediction
    cap = cv2.VideoCapture(0)  # Open camera
    while True:
        success, frame = cap.read()
        if not success:
            break

        # MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Extract landmark features
            landmarks = results.pose_landmarks.landmark
            ref_landmark = landmarks[11]  # Reference point
            ref_x, ref_y, ref_z = ref_landmark.x, ref_landmark.y, ref_landmark.z

            # Calculate relative positions
            feature_vector = []
            for lm in landmarks[:13]:  # Only take the first 13 landmarks
                feature_vector.extend([lm.x - ref_x, lm.y - ref_y, lm.z - ref_z])
            feature_vector = np.array(feature_vector, dtype=np.float32)

            # Convert features to PyTorch Tensor and classify
            feature_tensor = torch.tensor(feature_vector).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                prediction = model(feature_tensor).item()

            # Determine classification result
            label = "Good Posture" if prediction > 0.5 else "Bad Posture"
            last_prediction = prediction
            color = (0, 255, 0) if label == "Good Posture" else (0, 0, 255)

            # Display classification result
            cv2.putText(frame, f"{label} ({prediction:.2f})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Record data if in recording mode
            if recording_normal or recording_abnormal:
                data_folder = "data/normal" if recording_normal else "data/abnormal"
                filename = os.path.join(data_folder, f"data_{normal_data_count if recording_normal else abnormal_data_count}.npy")
                np.save(filename, feature_vector)
                if recording_normal:
                    normal_data_count += 1
                else:
                    abnormal_data_count += 1
        else:
            last_prediction = None

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    print("normal_data_count",normal_data_count)
    print("abnormal_data_count",abnormal_data_count)
    return render_template('index.html', normal_count=normal_data_count, abnormal_count=abnormal_data_count)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset', methods=['POST'])
def reset():
    global normal_data_count, abnormal_data_count, model
    model = PostureClassifier(input_dim)
    if os.path.exists("posture_classifier.pth"):
        os.remove("posture_classifier.pth")
    shutil.rmtree('data')
    os.makedirs('data/normal', exist_ok=True)
    os.makedirs('data/abnormal', exist_ok=True)
    normal_data_count = 0
    abnormal_data_count = 0
    return jsonify(success=True)

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording_normal, recording_abnormal
    data_type = request.json.get('type')
    if data_type == 'normal':
        recording_normal = True
    elif data_type == 'abnormal':
        recording_abnormal = True
    return jsonify(success=True)

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording_normal, recording_abnormal
    recording_normal = False
    recording_abnormal = False
    return jsonify(success=True)

@app.route('/file_counts', methods=['GET'])
def file_counts():
    normal_count = len(os.listdir('data/normal'))
    abnormal_count = len(os.listdir('data/abnormal'))
    return jsonify(normal=normal_count, abnormal=abnormal_count)

@app.route('/start_training', methods=['POST'])
def start_training():
    global model
    # Assuming you have the function to train the model
    acc, model = train.train_model()
    return jsonify(success=True, accuracy=acc.item())

@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    global last_prediction
    if last_prediction is None:
        return jsonify(success=False,prediction=None)
    return jsonify(success=True, prediction=last_prediction)
    

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)