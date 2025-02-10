import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
from tensorflow.keras.models import load_model
import json
from collections import defaultdict

app = FastAPI()

# Load the trained LSTM model once
model = load_model('action.h5')

# Initialize MediaPipe Holistic Model once
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

actions = ['NO_SIGN', 'A', 'B', 'C']  # Define sign labels

# Dictionary to store per-user sequences
user_sequences = defaultdict(list)

# Function to extract keypoints from MediaPipe
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    user_sequences[client_id] = []  # Initialize sequence for the user

    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_bytes()
            img_data = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

            # Convert to RGB & detect landmarks
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)

            # Store the latest 30 frames per user
            user_sequences[client_id].append(keypoints)
            user_sequences[client_id] = user_sequences[client_id][-30:]  # Keep only last 30 frames

            # Perform prediction if we have 30 frames
            if len(user_sequences[client_id]) == 30:
                res = model.predict(np.expand_dims(user_sequences[client_id], axis=0))[0]
                predicted_sign = actions[np.argmax(res)]

                # Send prediction result only to the respective user
                await websocket.send_text(json.dumps({"predicted_sign": predicted_sign}))

    except WebSocketDisconnect:
        print(f"User {client_id} disconnected")
        del user_sequences[client_id]  # Remove user's data when they disconnect

    finally:
        print(f"Closing WebSocket connection for {client_id}")
