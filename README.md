# 🎓 Automated Attendance System using Facial Recognition

**Helwan University**
Faculty of Computers and Artificial Intelligence
Computer Science & Artificial Intelligence & Information Technology Departments

![System Demo GIF](assets/images/demo.gif)

---

## 🌟 Project Overview

An **Automated Attendance System** using **facial recognition** for students and employees.
The system provides **real-time attendance tracking**, making the process **fast, accurate, and secure**.

---

## ⚡ Key Features

* 🧑‍🎓 Automatic recognition of students and employees
* ⏱ Real-time attendance logging
* 📊 Export attendance data to **Excel**
* 🔒 Secure and reliable system
* 🌐 User-friendly **web interface** built with Flask

---

## 🛠 Technologies Used

* 🐍 **Python 3.x** ![Python Badge](https://img.shields.io/badge/Python-3776AB?style=flat\&logo=python\&logoColor=white)
* 🌐 **Flask** ![Flask Badge](https://img.shields.io/badge/Flask-000000?style=flat\&logo=flask\&logoColor=white)
* 🖥 **OpenCV & face-recognition** ![OpenCV Badge](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat\&logo=opencv\&logoColor=white)
* 🔢 **NumPy** ![NumPy Badge](https://img.shields.io/badge/NumPy-013243?style=flat\&logo=numpy\&logoColor=white)
* 🗄 **Pandas & OpenPyXL** ![Pandas Badge](https://img.shields.io/badge/Pandas-150458?style=flat\&logo=pandas\&logoColor=white)

---

## 💾 Installation Guide

1. Clone the project:

   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project folder:

   ```bash
   cd Graduation\ project-1/Graduation\ project/emp_att
   ```
3. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / Mac
   venv\Scripts\activate      # Windows
   ```
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:

   ```bash
   python app.py
   ```
6. Open in browser:

   ```
   http://localhost:5000
   ```
7. Add images in `dataset/` for recognition

---

## 📂 Project Structure

```
Graduation project/
│
├─ Student attend/       # Student module
│   ├─ app.py
│   ├─ dataset/
│   ├─ face_db.pkl
│   └─ templates/        # dashboard.html, index.html
│
├─ emp_att/              # Employee module
│   ├─ app.py
│   ├─ app2.py
│   ├─ dataset/
│   ├─ face_db.pkl
│   ├─ requirements.txt
│   └─ templates/        # base.html, dashboard.html, index.html
│
├─ 55.png                # Screenshot
├─ video_2025-06-15_05-40-54.mp4  # Demo video
├─ FCAI Helwan GP Documentation-1.pdf
└─ Facial Recognition Attendance System.pptx
```

---

## 🏆 Author

**Amr Eid**
Helwan University – Faculty of Computers and AI

---

## 🎬 Demo Video

[![Watch Demo](video_2025-06-15_05-40-54.mp4)](video_2025-06-15_05-40-54.mp4)
