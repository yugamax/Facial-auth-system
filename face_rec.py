import os
import cv2
import numpy as np
import face_recognition

data_location = 'dataset'

def register_user(username, img1, img2):
    user_folder = os.path.join(data_location, username)
    os.makedirs(user_folder, exist_ok=True)

    def get_encoding(img_path):
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            raise ValueError(f"No face found in {img_path}")
        return encodings[0]

    enc1 = get_encoding(img1)
    enc2 = get_encoding(img2)

    avg_encoding = (enc1 + enc2) / 2

    cv2.imwrite(os.path.join(user_folder, 'face1.jpg'), cv2.imread(img1))
    cv2.imwrite(os.path.join(user_folder, 'face2.jpg'), cv2.imread(img2))
    np.save(os.path.join(user_folder, 'encoding.npy'), avg_encoding)
    print(f"{username} is registered successfully.")

def verify_user(username, live_img_p, tolerance=0.5):
    user_folder = os.path.join(data_location, username)
    encoding_file = os.path.join(user_folder, 'encoding.npy')

    if not os.path.exists(encoding_file):
        print(f"No encoding found for user, {username}")
        return False

    known_encoding = np.load(encoding_file)

    image = face_recognition.load_image_file(live_img_p)
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