import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado (asegúrate de tener tu modelo en el mismo directorio)
model = load_model('sign_language_model.h5')

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

def recognize_gesture_from_video():
    """Reconocer el gesto de la mano en un flujo de video en tiempo real"""
    cap = cv2.VideoCapture(0)  # Abrir la cámara (0 es el índice predeterminado)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
                landmarks = landmarks.flatten().reshape(1, -1)
                
                # Predicción con el modelo
                prediction = model.predict(landmarks)
                gesture = np.argmax(prediction, axis=1)
                gesture_name = str(gesture[0])

                # Dibujar las manos detectadas
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Mostrar el nombre del gesto detectado en la pantalla
                cv2.putText(frame, f'Gesture: {gesture_name}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mostrar la imagen en una ventana
        cv2.imshow("Sign Language Recognition", frame)

        # Presiona 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_gesture_from_video()
