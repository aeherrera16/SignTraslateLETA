# record_dataset.py
import cv2
import numpy as np
import os
import mediapipe as mp
from utils.landmark_extractor import extract_hand_landmarks

# Configura aquí tus clases
labels = ['A', 'B', 'C', 'D', 'E', 'F']  # Puedes cambiar esto
num_samples_per_class = 100  # Por ejemplo, 100 muestras por letra

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

X_data = []
y_data = []

cap = cv2.VideoCapture(0)

for idx, label in enumerate(labels):
    print(f"✋ Coloca la seña para la letra '{label}'. Comienza en 5 segundos...")
    cv2.waitKey(5000)

    count = 0
    while count < num_samples_per_class:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = extract_hand_landmarks(hand_landmarks)
                if len(landmarks) == 63:
                    X_data.append(landmarks)
                    y_data.append(idx)
                    count += 1
                    print(f"{label}: {count}/{num_samples_per_class}")

        cv2.putText(image, f"Letra: {label} ({count}/{num_samples_per_class})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Recolectando dataset", image)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Guardar en archivos
os.makedirs("data", exist_ok=True)
np.save("data/X_landmarks.npy", np.array(X_data))
np.save("data/y_labels.npy", np.array(y_data))
print("✅ Dataset guardado en /data")

cap.release()
cv2.destroyAllWindows()
