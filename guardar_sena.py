import pyodbc
import cv2
import os
from datetime import datetime

# Configura tu conexión a SQL Server
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=localhost;'               # Cambia si no es localhost
    'DATABASE=LenguaSeñasDB;'
    'UID=sa;'                 # Cambia esto
    'PWD=123;'             # Cambia esto
)
cursor = conn.cursor()

# Ruta donde guardarás los videos
video_dir = "videos"
os.makedirs(video_dir, exist_ok=True)

# Pedir nombre de la seña
nombre = input("Nombre de la seña: ")

# Grabar video con la cámara
nombre_archivo = f"{nombre}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
ruta_completa = os.path.join(video_dir, nombre_archivo)

cap = cv2.VideoCapture(0)
width = int(cap.get(3))
height = int(cap.get(4))
out = cv2.VideoWriter(ruta_completa, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

print("Grabando... Presiona 'q' para detener")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    cv2.imshow("Grabación", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Guardar la info en SQL Server
query = "INSERT INTO Señas (nombre, ruta_video) VALUES (?, ?)"
cursor.execute(query, (nombre, ruta_completa))
conn.commit()
conn.close()

print(f"✅ Seña '{nombre}' guardada en la base de datos con video en {ruta_completa}")
