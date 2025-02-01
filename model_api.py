# server.py
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import json

app = FastAPI()

# Load the trained LSTM model
model = load_model('action.h5')

# MediaPipe for hand tracking
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

actions = ['NO_SIGN', 'A', 'B', 'C']  # Define your actions here
label_map = {label: num for num, label in enumerate(actions)}

# Function to extract keypoints from MediaPipe
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sequence = []
    while True:
        try:
            # Receive the binary data (frame)
            data = await websocket.receive_bytes()  # Use receive_bytes instead of receive_text

            # Convert the received binary data to an image
            img_data = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)  # Decode the image

            # Make predictions
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]  # Keep only the last 30 frames

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predicted_sign = actions[np.argmax(res)]
                    await websocket.send_text(json.dumps({"predicted_sign": predicted_sign}))
        except WebSocketDisconnect:
            print("User disconnected")
            break

# Helper function for MediaPipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results
