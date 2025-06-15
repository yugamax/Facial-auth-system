from fastapi import FastAPI, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
from db_init import SessionLocal, engine
from db_handling import FaceEncoding, Base
import numpy as np
import face_recognition
import uvicorn
import os

Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_face_encoding(file: UploadFile):
    image = face_recognition.load_image_file(file.file)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise ValueError("No face found in the image")
    return encodings[0]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/ping")
def ping():
    return {"message": "Server is running"}

@app.post("/register/")
async def register_user(
    username: str = Form(...),
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        enc1 = get_face_encoding(image1)
        enc2 = get_face_encoding(image2)
        avg_encoding = ((enc1 + enc2) / 2).tolist()

        face_check = db.query(FaceEncoding).filter(FaceEncoding.username == username).first()
        if face_check:
            face_check.encoding = avg_encoding
        else:
            face_check = FaceEncoding(username=username, encoding=avg_encoding)
            db.add(face_check)
        db.commit()

        return {"message": f"{username} registered successfully."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/verify/")
async def verify_user(
    username: str = Form(...),
    live_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    face_check = db.query(FaceEncoding).filter(FaceEncoding.username == username).first()
    if not face_check:
        return {"verified": False, "message": "User not found"}

    try:
        known_encoding = np.array(face_check.encoding)
        live_encoding = get_face_encoding(live_image)
        distance = face_recognition.face_distance([known_encoding], live_encoding)[0]

        tolerance = 0.6
        confidence = max(0.0, 1.0 - distance) * 100

        return {
            "verified": True if distance < tolerance else False,
            "confidence": f"{confidence:.2f}%",
            "message": "Access granted" if distance < tolerance else "Access denied"
        }
    except Exception as e:
        return {"verified": False, "error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)