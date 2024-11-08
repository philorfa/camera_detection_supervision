from ultralytics import YOLO
import os
from google.cloud import vision
import cv2
from src.utils.timestamps import get_date_time_dict
from src.utils.plate_recognition.plate_utils import (updated_car_status_rgb,
                                                     send_car_update_rgb, 
                                                     process_response,
                                                     xai_plate_recognition)
from src.utils.plate_recognition.plate_recognition import recognize_plates
from src.utils.logger import log
from src.core.settings import get_settings
from src.utils.plate_recognition.car_plate import Car_RGB
from src.utils.video import return_frame_from_video, crop_detections

setting = get_settings()
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"] = setting.GOOGLE_APPLICATION_CREDENTIALS


def analyze_rgb_frames(visualize_detection: bool,
                       visualize_xai: bool) -> None:
    """Analyze RGB frames from a video for vehicle and license plate
    recognition.

    This function initializes the necessary models, reads frames from a
    specified video, and processes each frame to detect vehicles and license
    plates. It logs the detection results and optionally visualizes the frames
    with detected bounding boxes.

    Args:
        visualize (bool): If True, display the detected frames with bounding
        boxes.

    Returns:
        None
    """

    log.info("Starting Initialization ...")
    log.info("Setting up models ...")

    # Load object detection models
    car_model = YOLO(f"{os.getcwd()}/src/ml_models/car_detector8n.pt")
    
    plate_model = YOLO(
        f"{os.getcwd()}/src/ml_models/license_plate_detector.pt")

    # Load OCR model
    ocr_model = vision.ImageAnnotatorClient()

    # Load video
    rgb_video = "rgb.mp4"
    rgb_video_path = os.path.join("data/rgb_data", str(5), rgb_video)
    cap_rgb = cv2.VideoCapture(rgb_video_path)
    total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))

    log.info("Reading frames ...")

    # Start reading frames
    for i in range(total_frames - 400, total_frames):
        ret, frame = return_frame_from_video(cap_rgb, i)
        if ret:
            frame_info = get_date_time_dict()

            detections, ids_detected = recognize_plates(
                frame, car_model, plate_model, ocr_model)
            frame_info.update({"detections": detections})

            cars_to_delete = []

            for car_id, car in Car_RGB.cars_dict.items():
                cars_to_delete, ids_detected, updated = updated_car_status_rgb(
                    car_id, car, detections, ids_detected, cars_to_delete)

                if updated:
                    xai_image = xai_plate_recognition(plate_image=crop_detections(frame, car.car_coords),
                                                      plate_model=plate_model)
                    response = send_car_update_rgb(frame, xai_image, car)
                    process_response(component='plate',
                                     response=response.json(),
                                     visualize_detection=visualize_detection,
                                     visualize_xai=visualize_xai)

            for car_id in cars_to_delete:
                del Car_RGB.cars_dict[car_id]

            for first_seen_id in ids_detected:
                car_detected = detections.get(first_seen_id, None)

                if car_detected and car_detected["text"] is not None:
                    car = Car_RGB.create_car(first_seen_id, car_detected)

                    xai_image = xai_plate_recognition(plate_image=crop_detections(frame, car.car_coords),
                                                      plate_model=plate_model)
                    
                    response = send_car_update_rgb(frame, xai_image, car)
                    process_response(component='plate',
                                     response=response.json(),
                                     visualize_detection=visualize_detection,
                                     visualize_xai=visualize_xai)
