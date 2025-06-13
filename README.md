# 🚶‍♂️ Crowd Person Detection App

A simple and interactive **Streamlit web app** that uses the **YOLOv8 deep learning model** to detect and count people in uploaded crowd images.

Built with 💻 **Python**, 🧠 **Ultralytics YOLOv8**, and 🎨 **Streamlit**, this app offers an easy way to estimate crowd density using computer vision.

---

## 📸 How It Works

1. The user uploads an image (JPEG, PNG).
2. The app loads the YOLOv8 model pre-trained on the COCO dataset.
3. It detects all instances of the "person" class (class ID 0).
4. The app returns:
   - ✅ Number of people detected
   - 🖼 Annotated image with bounding boxes around each person

---

## 🌟 Features

- 🧠 Powered by **YOLOv8** (`yolov8n.pt`)
- ✅ Real-time object detection
- 🎯 Detects only **humans** in the scene
- 📊 Crowd count output
- 🖼 Annotated result image
- 🌐 Simple **web interface** via Streamlit

---

## 🧰 Technologies Used

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- OpenCV
- Pillow (PIL)
- NumPy

---

## ⚙️ Installation & Setup

### ✅ 1. Clone the Repository

- git clone https://github.com/venkatakoteswarao/crowd-person-detector.git
- cd crowd-person-detector
