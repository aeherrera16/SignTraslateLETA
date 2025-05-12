import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Simula 1000 muestras con 63 features (landmarks) y 10 clases (puedes ajustar)
X = np.random.rand(1000, 63).astype('float32')
y = np.random.randint(0, 10, size=(1000,))
y_cat = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # cambia a tu número real de clases si ya lo sabes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Guardar el modelo
model.save("sign_language_model_landmarks.h5")
