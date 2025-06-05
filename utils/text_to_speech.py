import pyttsx3
import threading

# Inicializar el engine solo una vez
engine = pyttsx3.init()
engine.setProperty('rate', 100)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

lock = threading.Lock()  # Evita que hablen muchas voces juntas

def speak_text(text):
    def run():
        with lock:
            engine.say(text)
            engine.runAndWait()

    threading.Thread(target=run, daemon=True).start()
