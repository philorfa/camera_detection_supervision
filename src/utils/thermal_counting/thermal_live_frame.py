from ultralytics import YOLO
import os
import cv2
from src.utils.video import return_frame_from_video
from src.utils.timestamps import get_date_time_dict
from src.utils.thermal_counting.thermal_people_counting import thremal_count
from src.utils.thermal_counting.thermal_utils import (updated_car_status_thermal, 
                                                      send_car_update_thermal,
                                                      process_response,
                                                      xai_people_count)
from src.utils.logger import log
from src.core.settings import get_settings
from src.utils.thermal_counting.car_thermal import Car_Thermal
from src.utils.video import crop_detections

settings = get_settings()


def analyze_thermal_frames(visualize_detection: bool,
                           visualize_xai: bool) -> None:
    """Analyze thermal video frames for car and people detection.

    This function initializes models for car and people detection,
    reads frames from a thermal video, and processes each frame to
    detect cars and people. It updates the status of detected cars
    and sends updates as necessary.

    Args:
        visualize (bool): If True, visualize the detected frames.

    Returns:
        None
    """

    log.info("Starting Initialization ...")
    log.info("Setting up models ...")

    # load car detection model
    car_model_directory = f"{os.getcwd()}/src/ml_models/thermal_car.pt"
    car_model = YOLO(car_model_directory)

    # load people detection model
    people_model_directory = f"{os.getcwd()}/src/ml_models/thermal_peoplev8.pt"
    people_model = YOLO(people_model_directory)

    # load video
    thermal_video = "thermal.mp4"
    thermal_video_path = os.path.join(f"{os.getcwd()}/data/thermal_data",
                                      str(5), thermal_video)

    cap_thermal = cv2.VideoCapture(thermal_video_path)
    total_frames = int(cap_thermal.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Reading frames ...")

    # Start reading frames
    for i in range(total_frames - 400, total_frames):
        ret, frame = return_frame_from_video(cap_thermal, i)
        if ret:
            frame_info = get_date_time_dict()

            detections, ids_detected = thremal_count(frame, car_model,
                                                     people_model, 0.3)

            frame_info.update({"detections": detections})
            cars_to_delete = []
            for car_id, car in Car_Thermal.cars_dict.items():
                (cars_to_delete, ids_detected,
                 updated) = updated_car_status_thermal(car_id, car, detections,
                                                       ids_detected,
                                                       cars_to_delete)
                if updated:
                    xai_image = xai_people_count(car_image=crop_detections(frame, car.car_coords),
                                                 model_dir=people_model_directory)
                    
                    response = send_car_update_thermal(frame, xai_image, car)

                    process_response(component='people',
                                     response=response.json(),
                                     visualize_detection=visualize_detection,
                                     visualize_xai=visualize_xai)
                    
            for car_id in cars_to_delete:
                del Car_Thermal.cars_dict[car_id]

            for first_seen_ids in ids_detected:
                car_detected = detections.get(first_seen_ids, None)

                if car_detected["people"] is not None:
                    car = Car_Thermal.create_car(first_seen_ids, car_detected)
                    xai_image = xai_people_count(car_image=crop_detections(frame, car.car_coords),
                                                 model_dir=people_model_directory)
                    response = send_car_update_thermal(frame, xai_image, car)
                    process_response(component='people',
                                     response=response.json(),
                                     visualize_detection=visualize_detection,
                                     visualize_xai=visualize_xai)
