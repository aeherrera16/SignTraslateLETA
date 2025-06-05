# app.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
from tensorflow.keras.models import load_model
from utils.landmark_extractor import extract_hand_landmarks
from utils.text_to_speech import speak_text  # <-- ahora sí bien importado

# Cargar modelo y etiquetas
model = load_model("sign_language_model_landmarks.h5")
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Iniciar cámara
cap = cv2.VideoCapture(0)
prev_prediction = ''
last_spoken_time = 0  # Guarda el tiempo de la última vez que habló
speak_interval = 0.3  # Segundos mínimos entre voces

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = extract_hand_landmarks(hand_landmarks)

            if len(landmarks) == 63:
                prediction = model.predict(np.array([landmarks]), verbose=0)
                predicted_index = np.argmax(prediction)
                predicted_label = labels[predicted_index]

                cv2.putText(image, predicted_label, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                # Solo hablar si es una nueva predicción y pasó suficiente tiempo
                current_time = time.time()
                if predicted_label != prev_prediction and (current_time - last_spoken_time) > speak_interval:
                    speak_text(predicted_label)
                    prev_prediction = predicted_label
                    last_spoken_time = current_time

    cv2.imshow("Sign Language Recognition", image)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc para salir
        break

cap.release()
cv2.destroyAllWindows()
