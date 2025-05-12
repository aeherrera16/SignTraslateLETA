import sqlite3

def connect_db():
    conn = sqlite3.connect('sign_language.db')
    return conn

def create_table():
    conn = connect_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS gestures
                 (id INTEGER PRIMARY KEY, gesture TEXT, description TEXT)''')
    conn.commit()
    conn.close()

def insert_gesture(gesture, description):
    conn = connect_db()
    c = conn.cursor()
    c.execute("INSERT INTO gestures (gesture, description) VALUES (?, ?)", (gesture, description))
    conn.commit()
    conn.close()

def get_description(gesture):
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT description FROM gestures WHERE gesture=?", (gesture,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 'Desconocido'
