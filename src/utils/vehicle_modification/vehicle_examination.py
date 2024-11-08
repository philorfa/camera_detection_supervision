
from ultralytics import YOLO
from src.utils.logger import log
import os
import cv2
from src.utils.video import return_frame_from_video
from src.utils.timestamps import get_date_time_dict
from src.utils.vehicle_modification.car_recognition import recognize_cars
from src.utils.vehicle_modification.car_modified import Car_Properties
from src.utils.vehicle_modification.vehicle_utils import (updated_car_images_rgb, 
                                                          identify_properties,
                                                          send_car_properties_rgb,
                                                          process_response_modifications)


def examine_rgb_frames(visualize: bool) -> None:

    """
    Processes RGB video frames to detect and analyze vehicles, identifying any modifications.
    
    This function initializes object detection models, reads frames from an RGB video, 
    detects vehicles, and analyzes each detected vehicle for any modifications. If 
    a vehicle is identified, its properties are analyzed, sent for processing, and 
    modifications are identified and logged.

    Args:
        visualize (bool): Flag indicating whether to visualize processing results.

    Returns:
        None
    """
    
    log.info("Starting Initialization ...")
    log.info("Setting up models ...")

    # Load object detection models
    car_model = YOLO(f"{os.getcwd()}/src/ml_models/car_detector8n.pt")


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
            detections, ids_detected = recognize_cars(frame, car_model)
        
            frame_info.update({"detections": detections})

            cars_to_delete = []
            # Car_Properties.print_all_cars()
            
            # Update cars color and type from cars that are already found
            for car_id, car in Car_Properties.cars_dict.items():

                cars_to_delete, ids_detected, ready_for_identification = updated_car_images_rgb(
                    car_id, car, detections, ids_detected, cars_to_delete)
                
                if ready_for_identification:
                    car = identify_properties(car)
                    response = send_car_properties_rgb(car)
                    process_response_modifications(component='vehicle modification',
                                                   response=response.json(),
                                                   visualize=visualize)
                    
            # Delete cars that are not detected anymore
            for car_id in cars_to_delete:
                del Car_Properties.cars_dict[car_id]
            
            # Create first seen cars object
            for first_seen_id in ids_detected:
                car_detected = detections.get(first_seen_id, None)

                if car_detected is not None:
                    car = Car_Properties.create_car(first_seen_id, car_detected)

    return