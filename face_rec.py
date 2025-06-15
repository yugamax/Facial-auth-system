from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
import psycopg2
import face_recognition
from typing import List

conn = psycopg2.connect(
    host="your-neon-host",
    database="your-db",
    user="your-user",
    password="your-password",
    port="5432"
)
cursor = conn.cursor()

app = FastAPI()

def get_face_encoding(file: UploadFile):
    image = face_recognition.load_image_file(file.file)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise ValueError("No face found in the image")
    return encodings[0]

@app.post("/register/")
async def register_user(
    username: str = Form(...),
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    try:
        enc1 = get_face_encoding(image1)
        enc2 = get_face_encoding(image2)
        avg_encoding = ((enc1 + enc2) / 2).tolist()

        cursor.execute("""
            INSERT INTO face_encodings (username, encoding)
            VALUES (%s, %s)
            ON CONFLICT (username) DO UPDATE SET encoding = EXCLUDED.encoding;
        """, (username, avg_encoding))
        conn.commit()

        return {"message": f"{username} registered successfully."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/verify/")
async def verify_user(
    username: str = Form(...),
    live_image: UploadFile = File(...),
    tolerance: float = Form(0.5)
):
    cursor.execute("SELECT encoding FROM face_encodings WHERE username = %s;", (username,))
    result = cursor.fetchone()

    if not result:
        return {"verified": False, "message": f"No encoding found for {username}"}

    known_encoding = np.array(result[0])

    try:
        live_encoding = get_face_encoding(live_image)
        distance = face_recognition.face_distance([known_encoding], live_encoding)[0]

        print(f"Distance: {distance}, Tolerance: {tolerance}")
        confidence = max(0.0, 1.0 - distance) * 100
        return {
            "verified": distance < tolerance,
            "confidence": f"{confidence:.2f}%",
            "message": "Face match! Access granted." if distance < tolerance else "Face mismatch. Access denied."
        }

    except Exception as e:
        return {"verified": False, "error": str(e)}