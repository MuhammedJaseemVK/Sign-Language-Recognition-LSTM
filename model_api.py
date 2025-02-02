import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
from tensorflow.keras.models import load_model
import json

app = FastAPI()

# Load the trained LSTM model once
model = load_model('action.h5')

# Initialize MediaPipe Holistic Model once (outside WebSocket loop for efficiency)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

actions = ['NO_SIGN', 'A', 'B', 'C']  # Define your sign labels
sequence = []  # Stores last 30 frames

# Function to extract keypoints from MediaPipe
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global sequence  # Use the same sequence globally

    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_bytes()
            img_data = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)  # Convert binary to image

            # Convert to RGB & detect landmarks
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)

            # Update sequence with the latest frame
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep only last 30 frames

            # Perform prediction if we have 30 frames
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_sign = actions[np.argmax(res)]

                # Send prediction result to client
                await websocket.send_text(json.dumps({"predicted_sign": predicted_sign}))

    except WebSocketDisconnect:
        print("User disconnected")

    finally:
        print("Closing WebSocket connection")
