# extraer_landmarks_video.py
import cv2
import mediapipe as mp
import csv
import os
from landmark_extractor import extract_hand_landmarks  # Importa la función

# Ruta del video que quieres procesar
VIDEO_PATH = "videos/Hola.mp4"  # Cambia por tu video
OUTPUT_CSV = "dataset/Hola.csv"  # Nombre de la seña

# Crear carpeta si no existe
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = extract_hand_landmarks(hand_landmarks)  # Usar la función importada
                if len(landmarks) == 63:
                    writer.writerow(landmarks)
                    frame_count += 1

cap.release()
hands.close()
print(f"✅ Landmarks guardados en {OUTPUT_CSV} ({frame_count} muestras)")
