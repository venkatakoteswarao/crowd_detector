import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os

# Function to detect people in the uploaded image
def detect_people(image):
    # Load YOLOv8 model (pretrained on COCO dataset)
    model = YOLO('yolov8n.pt')  # 'n' stands for nano; you can use 's', 'm', 'l', or 'x' for better accuracy

    # Convert PIL Image to OpenCV format
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Perform detection
    results = model(image_bgr)

    # Filter detections for class 'person' (class ID 0 in COCO dataset)
    person_count = sum([1 for obj in results[0].boxes if int(obj.cls) == 0])

    # Annotate the image
    annotated_img = results[0].plot()

    return person_count, annotated_img

# Streamlit app
def main():
    st.title("🚶‍♂️ Crowd Person Detection App")
    st.write("Upload an image, and the model will detect the number of people in the crowd!")

    # File uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image
        with st.spinner("Processing... Please wait!"):
            person_count, annotated_img = detect_people(image)

        # Display results
        st.success(f"Number of people detected: {person_count}")
        st.image(annotated_img, caption="Annotated Image", use_column_width=True)

# Run the app
if __name__ == "__main__":
    main()
