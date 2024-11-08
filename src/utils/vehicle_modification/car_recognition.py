from typing import Tuple, Dict, List, Any
from src.utils.video import crop_detections
from src.utils.vehicle_modification.vehicle_utils import detect_car

def recognize_cars(image: Any, car_model: Any) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    """
    Detects cars in an image using the specified model and returns detection information.
    
    Args:
        image (Any): The input image in which cars will be detected.
        car_model (Any): The pre-trained model used for detecting cars.

    Returns:
        Tuple[Dict[int, Dict[str, Any]], List[int]]:
            - A dictionary containing information for each detected car, keyed by car ID.
            - A list of IDs for the detected cars.
    """
    car_detections = detect_car(image, car_model, conf=0.4)
    info, ids_detected = extract_car_info(image, car_detections)
    return (info, ids_detected)


def extract_car_info(image: Any, car_detections: Any) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    """
    Extracts relevant information from detected cars.

    Args:
        image (Any): The input image containing detected cars.
        car_detections (Any): The detected car data from the model.

    Returns:
        Tuple[Dict[int, Dict[str, Any]], List[int]]:
            - A dictionary containing the coordinates, confidence, original image,
              and cropped image for each detected car.
            - A list of IDs for the detected cars.
    """
    info = {}
    ids_detected = []

    for car in car_detections.boxes.data.tolist():
        car_id = int(car[4])
        ids_detected.append(car_id)
        zoomed_in_car_rgb = crop_detections(image, car)

        info[car_id] = {
            "car_coords": car[:4],
            "car_confidence": car[5],
            "original_image": image,
            "cropped_image": zoomed_in_car_rgb
        }

    return info, ids_detected
