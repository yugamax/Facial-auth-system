
---

# 🧠 Facial-auth-system

A ⚡ FastAPI-based facial authentication system that enables 👤 user registration and identity verification using 🧬 face recognition and a 🗃️ PostgreSQL database.

---

## ✨ Features

* 👥 User registration using two facial images
* 📸 Face encoding with `face_recognition` (Dlib)
* 🧾 Identity verification with live image input
* ✅ Confidence-based verification decision
* 🛠️ SQLAlchemy ORM for database interaction
* 🌐 REST API built with FastAPI

---

## 📁 Folder Structure

```
Facial-auth-system/
├── face_rec.py           # 🧠 FastAPI app with register and verify endpoints
├── db_init.py            # 🔧 SQLAlchemy engine and session setup
├── db_handling.py        # 📦 Database model for face encodings
├── requirements.txt      # 📜 Python dependencies
├── .env                  # 🔐 Environment variable for DB URL
```

---

## ⚙️ Setup Instructions

1. 📥 Clone the repository
2. 🧪 Set up a virtual environment
3. 📦 Install dependencies using `requirements.txt`
4. 📝 Add a `.env` file with the `DATABASE_URL`
5. 🚀 Run the FastAPI app using Uvicorn

---

## 🔌 API Endpoints

* `GET /ping` – ✅ Health check endpoint
* `POST /register/` – 👤 Register a new user with two images
* `POST /verify/` – 🔍 Verify identity using a live image

---

## 🔍 Technical Details

* 🧬 Face encodings are generated using Dlib via `face_recognition`
* 🧑‍🤝‍🧑 Two images are averaged into one encoding per user
* 🔄 Verification compares the stored encoding with a new one
* 📉 Confidence score calculated via Euclidean distance
* 🎯 Verification passes if distance < 0.6

---

## 🧰 Dependencies

* 🚀 FastAPI
* ⚡ Uvicorn
* 🧠 face\_recognition
* 🗃️ SQLAlchemy
* ➗ NumPy
* 🧪 python-dotenv

---

## 📝 Notes

* 📷 Use clear, front-facing images
* 🧪 Suitable for educational or prototype use

---