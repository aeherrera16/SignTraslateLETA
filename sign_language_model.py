import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Descargar el dataset de https://www.kaggle.com/datasets/datamunge/sign-language-mnist
# y colócalo en una carpeta llamada 'data'
TRAIN_CSV = 'data/sign_mnist_train/sign_mnist_train.csv'

# Cargar los datos
data = pd.read_csv(TRAIN_CSV)
labels = data['label'].values
images = data.drop('label', axis=1).values
images = images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Codificar las etiquetas
labels = to_categorical(labels)

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

# Crear el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(25, activation='softmax')  # 24 clases (A-Y sin J ni Z porque requieren movimiento)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Guardar el modelo
model.save('sign_language_model.h5')
print("✅ Modelo guardado como sign_language_model.h5")
