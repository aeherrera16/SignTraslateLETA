# model_landmarks.py
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Cargar datos
X = np.load("data/X_landmarks.npy")
y = np.load("data/y_labels.npy")

# Convertir etiquetas a categóricas
y_cat = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

# Crear modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Guardar modelo
model.save("sign_language_model_landmarks.h5")

# Guardar etiquetas en orden
labels = ['A', 'B', 'C', 'D', 'E', 'F']  # ⚠️ Ajusta según tus clases reales
with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)
