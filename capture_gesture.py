import cv2
import os
import numpy as np
import mediapipe as mp
from utils.landmark_extractor import extract_hand_landmarks
import csv

# Función para crear la carpeta para almacenar gestos
def create_gesture_folder(gesture_name):
    folder_path = f"dataset/{gesture_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Crea la carpeta si no existe
    return folder_path

# Función para guardar los landmarks extraídos en un archivo CSV
def save_landmarks_to_csv(landmarks, gesture_name):
    csv_filename = f"dataset/{gesture_name}/{gesture_name}_samples.csv"
    
    # Verificamos si el archivo ya existe para no agregar encabezado si ya existe
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Escribimos el encabezado si el archivo es nuevo
        if not file_exists:
            writer.writerow([f"x{i}" for i in range(1, 22)] + [f"y{i}" for i in range(1, 22)] + [f"z{i}" for i in range(1, 22)])

        # Guardamos los landmarks extraídos
        writer.writerow(landmarks)

# Iniciamos MediaPipe para el reconocimiento de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

# Solicitar al usuario el nombre del gesto (palabra)
gesture_name = input("Introduce el nombre del gesto (palabra): ")  # Ejemplo: "hola"
create_gesture_folder(gesture_name)  # Crear carpeta para la palabra

# Empezamos a capturar gestos
print("Grabando gesto en movimiento... Presiona 'Esc' para detener la grabación.")

# Inicia la captura en movimiento
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = extract_hand_landmarks(hand_landmarks)
            
            if len(landmarks) == 63:
                save_landmarks_to_csv(landmarks, gesture_name)  # Guardamos los landmarks con la etiqueta de la palabra

                # Dibujamos los landmarks en la pantalla
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Mostramos la imagen en tiempo real
    cv2.imshow("Captura de Seña en Movimiento", frame)
    
    # Espera que el usuario presione 'Esc' para salir
    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # ESC para salir
        print("Grabación detenida.")
        break
    
cap.release()
cv2.destroyAllWindows()
