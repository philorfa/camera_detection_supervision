from huggingface_hub import hf_hub_download
from ultralytics import YOLO

import cv2
import os

method = 2
video_name = '2130_2.mp4'
video_dir = os.path.join('acc_data/clips', video_name)

HUMAN = [0]
confidence = 0.3

if method == 1:
    thermal_model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-human-detection-thermal",
            filename="model.pt")
    thermal_model = YOLO(thermal_model_path)
    output_video_path = os.path.join('acc_data/model_output/yolo_thermal_arnabdhar', video_name)

if method == 2:
    model_path = hf_hub_download(
        repo_id="pitangent-ds/YOLOv8-human-detection-thermal",
        filename="model.pt"
    )
    thermal_model = YOLO(model_path)
    output_video_path = os.path.join('acc_data/model_output/yolo_thermal_pitangent', video_name)

if method == 3:
    thermal_model = YOLO("yolov8n.pt")
    output_video_path = os.path.join('acc_data/model_output/yolo_v8',
                                     video_name)

if __name__ == '__main__':
    cap = cv2.VideoCapture(video_dir)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        results = thermal_model(frame, classes=HUMAN, conf=confidence)

        # Draw bounding boxes on the frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = box.cls[0]
                label = f'{thermal_model.names[int(cls)]} {conf:.2f}'

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label near bounding box
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f'Output video saved to {output_video_path}')
