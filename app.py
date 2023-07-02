import cv2
import requests
import os
import streamlit as st
from streamlit.uploaded_file_manager import UploadedFile

from ultralytics import YOLO

def download_file(url, save_name):
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)

file_urls = [
    'https://www.dropbox.com/s/b5g97xo901zb3ds/pothole_example.jpg?dl=1',
    'https://www.dropbox.com/s/86uxlxxlm1iaexa/pothole_screenshot.png?dl=1',
    'https://www.dropbox.com/s/7sjfwncffg8xej2/video_7.mp4?dl=1'
]

for i, url in enumerate(file_urls):
    if 'mp4' in file_urls[i]:
        download_file(
            file_urls[i],
            f"video.mp4"
        )
    else:
        download_file(
            file_urls[i],
            f"image_{i}.jpg"
        )

model = YOLO('best.pt')

def show_preds_image(image_path):
    image = cv2.imread(image_path)
    outputs = model.predict(source=image_path)
    results = outputs[0].cpu().numpy()
    for i, det in enumerate(results.boxes.xyxy):
        cv2.rectangle(
            image,
            (int(det[0]), int(det[1])),
            (int(det[2]), int(det[3])),
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def show_preds_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_copy = frame.copy()
            outputs = model.predict(source=frame)
            results = outputs[0].cpu().numpy()
            for i, det in enumerate(results.boxes.xyxy):
                cv2.rectangle(
                    frame_copy,
                    (int(det[0]), int(det[1])),
                    (int(det[2]), int(det[3])),
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
            yield cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        else:
            break

def main():
    st.title("Pothole Detector")

    inference_type = st.sidebar.selectbox("Select Inference Type", ("Image", "Video"))

    if inference_type == "Image":
        st.header("Image Inference")
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            image_path = "uploaded_image.jpg"
            with open(image_path, "wb") as f:
                f.write(image_file.read())
            output_image = show_preds_image(image_path)
            st.image(output_image, caption="Output Image")

    elif inference_type == "Video":
        st.header("Video Inference")
        video_file = st.file_uploader("Upload a video", type=["mp4"])
        if video_file is not None:
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(video_file.read())
            output_images = show_preds_video(video_path)
            for output_image in output_images:
                st.image(output_image, caption="Output Image")

if __name__ == "__main__":
    main()
