
# 🚦 Smart Traffic Management System

An AI-powered solution for automated traffic rule enforcement using real-time video analytics. Developed using YOLOv8 and Python, this system detects red light violations, wrong-way driving, helmetless riders, and extracts license plates from video feeds — all through a user-friendly web interface.

## 📌 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## ✅ Features

- 🔴 **Red Light Violation Detection**
- ↩️ **Wrong Way Driving Detection**
- 🪖 **No Helmet Detection**
- 🔍 **License Plate Recognition**
- 📊 **Automatic Data Logging & Report Generation**
- 🌐 **Web Interface for Uploading & Viewing Results**
- 📦 **Modular and Scalable Design**

## 🛠 Tech Stack

| Component        | Technology                  |
|------------------|------------------------------|
| Backend          | Python 3.12, Flask           |
| Frontend         | HTML, CSS, JavaScript        |
| Object Detection | YOLOv8 (Ultralytics)         |
| Video Processing | OpenCV                       |
| Tracking         | SORT (Simple Online Realtime Tracking) |
| Data Storage     | CSV                          |

## 🧠 System Architecture

```
[ Video Upload ] 
      ↓
[ Flask Backend ]
      ↓
[ YOLOv8 Detection + OpenCV Processing ]
      ↓
[ Violation Detection Logic ]
      ↓
[ CSV Log + Web Interface Display ]
```

## ⚙️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/<your-username>/smart-traffic-management-system.git
cd smart-traffic-management-system
```

2. **Install dependencies manually:**

There is no `requirements.txt`. You will need to install the required packages individually, such as:

```bash
pip install flask opencv-python ultralytics
```

3. **Download YOLOv8 model:**

You can use a pretrained model from [Ultralytics](https://github.com/ultralytics/ultralytics).

## 🚀 Usage

1. **Open separate terminals** for each module (Red Light, Wrong Way, No Helmet, License Plate) and run their respective `app.py` files.

```bash
# Example for each terminal:
cd Helmet-Detection-System/
python app.py

cd wrong_way_detection/
python app.py

...
```

2. **Open `home.html`** in a web browser to access the interface.

3. **Upload a video** and select the desired detection mode.

4. **View Results:** Detected violations are logged in `records.csv` per module.

## 📁 Project Structure

```
├── red_light_violation/
│   └── app.py
├── wrong_way_detection/
│   └── app.py
├── helmet_detection/
│   └── app.py
├── license_plate_detection/
│   └── app.py
├── templates/
│   └── home.html
├── static/
├── uploads/
├── records.csv
```

## 📄 License

This project is for academic use under the University of Mumbai curriculum. For commercial or public use, please contact the authors.
