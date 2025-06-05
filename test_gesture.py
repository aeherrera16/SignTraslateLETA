import cv2
import numpy as np
import mediapipe as mp
import pickle  # Asegúrate de importar pickle
from tensorflow.keras.models import load_model
from utils.landmark_extractor import extract_hand_landmarks
from utils.text_to_speech import speak_text  # Asegúrate de tener esta función de TTS

# Función para cargar el modelo y las etiquetas
def load_model_and_labels():
    model = load_model("sign_language_model_landmarks.h5")  # Asegúrate de que el modelo está en el lugar correcto
    with open("labels.pkl", "rb") as f:
        labels = pickle.load(f)  # Carga las etiquetas de las palabras asociadas
    return model, labels

# Iniciar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Cargar modelo y etiquetas
model, labels = load_model_and_labels()

# Iniciar cámara
cap = cv2.VideoCapture(0)

print("Capturando gestos... Presiona 'Esc' para salir.")

prev_prediction = ''

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Procesar la imagen
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)
    
    # Si se detectan manos
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dibuja los puntos de los landmarks
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = extract_hand_landmarks(hand_landmarks)
            
            if len(landmarks) == 63:
                # Predicción de la seña
                prediction = model.predict(np.array([landmarks]), verbose=0)
                predicted_index = np.argmax(prediction)
                predicted_label = labels[predicted_index]
                
                # Mostrar el texto en la cámara
                cv2.putText(frame, predicted_label, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                # Si la predicción ha cambiado, decirla en voz alta
                if predicted_label != prev_prediction:
                    speak_text(predicted_label)  # Función de texto a voz
                    prev_prediction = predicted_label

    # Mostrar el video con la predicción y los landmarks
    cv2.imshow("Test Gesture", frame)
    
    # Esperar a que el usuario presione 'Esc' para salir
    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
