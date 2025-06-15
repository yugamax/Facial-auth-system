import os
import cv2
import numpy as np
import psycopg2
import face_recognition

conn = psycopg2.connect(
    host="your-neon-host",
    database="your-db",
    user="your-user",
    password="your-password",
    port="5432"
)
cursor = conn.cursor()

def get_encoding(img_path):
    image = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise ValueError(f"No face found in {img_path}")
    return encodings[0]

def register_user(username, img1_path, img2_path):
    enc1 = get_encoding(img1_path)
    enc2 = get_encoding(img2_path)
    avg_encoding = ((enc1 + enc2) / 2).tolist()

    with open(img1_path, 'rb') as f1, open(img2_path, 'rb') as f2:
        img1_bytes = f1.read()
        img2_bytes = f2.read()

    cursor.execute("""
        INSERT INTO face_encodings (username, encoding, img1, img2)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (username) DO UPDATE
        SET encoding = EXCLUDED.encoding,
            img1 = EXCLUDED.img1,
            img2 = EXCLUDED.img2;
    """, (username, avg_encoding, psycopg2.Binary(img1_bytes), psycopg2.Binary(img2_bytes)))

    conn.commit()
    print(f"{username} registered successfully.")

def verify_user(username, live_img_path, tolerance=0.5):
    cursor.execute("SELECT encoding FROM face_encodings WHERE username = %s;", (username,))
    result = cursor.fetchone()

    if not result:
        print(f"No encoding found for user {username}")
        return False

    known_encoding = np.array(result[0])

    image = face_recognition.load_image_file(live_img_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        print("No face found in live image.")
        return False

    live_encoding = encodings[0]
    distance = face_recognition.face_distance([known_encoding], live_encoding)[0]
    print(f"Distance: {distance:.2f}")

    if distance < tolerance:
        print("Face match! Transaction approved.")
        return True
    else:
        print("Face mismatch. Transaction denied.")
        return False

if __name__ == "__main__":
    register_user('adi', 'adi1.jpg', 'adi2.jpg')
    verify_user('adi', 'adi_test.jpg')