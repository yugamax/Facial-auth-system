from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
import psycopg2
import face_recognition
import os
from dotenv import load_dotenv
import json

load_dotenv()

app = FastAPI()

def get_connection():
    return psycopg2.connect(os.getenv("Database_URL"))

def get_face_encoding(file: UploadFile):
    image = face_recognition.load_image_file(file.file)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise ValueError("No face found in the image")
    return encodings[0]

@app.get("/ping")
def ping():
    return {"message": "server is running"}

@app.post("/register/")
async def register_user(
    username: str = Form(...),
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)):

    try:
        enc1 = get_face_encoding(image1)
        enc2 = get_face_encoding(image2)
        avg_encoding = ((enc1 + enc2) / 2).tolist()

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO face_encodings (username, encoding) 
            VALUES (%s, %s)
            ON CONFLICT (username) DO UPDATE 
            SET encoding = EXCLUDED.encoding;
        """, (username, json.dumps(avg_encoding)))
        conn.commit()
        cursor.close()
        conn.close()

        return {"message": f"{username} registered successfully."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/verify/")
async def verify_user(
    username: str = Form(...),
    live_image: UploadFile = File(...)):

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT encoding FROM face_encodings WHERE username = %s;", (username,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if not result:
            return {"verified": False, "message": f"No encoding found for {username}"}

        known_encoding = np.array(result[0])
        tolerance = 0.6
        live_encoding = get_face_encoding(live_image)
        distance = face_recognition.face_distance([known_encoding], live_encoding)[0]

        confidence = max(0.0, 1.0 - distance) * 100
        return {
            "verified": True if distance < tolerance else False,
            "confidence": f"{confidence:.2f}%",
            "message": "Face match! Access granted." if distance < tolerance else "Face mismatch. Access denied."
        }

    except Exception as e:
        return {"verified": False, "error": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)