import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Cargar el modelo
model = load_model("sign_language_model_landmarks.h5")
class_names = ["A", "B", "C", "D", "E"]  # Cambia según tus clases reales

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Captura desde la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip para vista espejo
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convertir a RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dibujar puntos
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer coordenadas (x, y, z) de cada uno de los 21 puntos
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            if len(data) == 63:
                prediction = model.predict(np.array([data]))  # (1, 63)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction)

                # Mostrar resultado
                cv2.putText(frame, f'{predicted_class} ({confidence:.2f})', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Sign Detection - Option B', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
