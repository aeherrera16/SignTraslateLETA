# model_landmarks.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Simula datos de entrada y salida
# X = [N muestras, 63 features]
# Y = [N etiquetas categóricas]
X = np.load("data/X_landmarks.npy")  # Debes tener este archivo
y = np.load("data/y_labels.npy")

y_cat = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(y_cat[0]), activation='softmax')  # N clases
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
model.save("sign_language_model_landmarks.h5")
