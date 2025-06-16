from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from db_init import SessionLocal, engine
from db_handling import FaceEncoding, Base
import numpy as np
import uvicorn
import os
import cv2
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FaceAnalysis(name='buffalo_s')
app.prepare(ctx_id=-1)

def read_image_from_upload(file: UploadFile):
    image_bytes = file.file.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

def get_face_embedding(file: UploadFile):
    img = read_image_from_upload(file)
    face_app = app
    faces = face_app.get(img)
    if not faces:
        raise ValueError("No face found in the image")
    return faces[0].embedding

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    print("App has started on Render!")

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
        emb1 = get_face_embedding(image1)
        emb2 = get_face_embedding(image2)
        avg_embedding = ((emb1 + emb2) / 2).tolist()

        face_check = db.query(FaceEncoding).filter(FaceEncoding.username == username).first()
        if face_check:
            face_check.encoding = avg_embedding
        else:
            face_check = FaceEncoding(username=username, encoding=avg_embedding)
            db.add(face_check)
        db.commit()

        return {"message": f"{username} registered successfully."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/verify/")
async def verify_user(
    username: str = Form(...),
    live_image: UploadFile = File(...),
    db: Session = Depends(get_db)):
    face_check = db.query(FaceEncoding).filter(FaceEncoding.username == username).first()
    if not face_check:
        return {"verified": False, "message": "User not found"}

    try:
        known_embedding = np.array(face_check.encoding)
        live_embedding = get_face_embedding(live_image)

        similarity = cosine_similarity([known_embedding], [live_embedding])[0][0]
        tolerance = 0.6
        confidence = similarity * 100

        return {
            "verified": True if similarity > tolerance else False,
            "confidence": f"{confidence:.2f}%",
            "message": "Access granted" if similarity > tolerance else "Access denied"
        }
    except Exception as e:
        return {"verified": False, "error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)