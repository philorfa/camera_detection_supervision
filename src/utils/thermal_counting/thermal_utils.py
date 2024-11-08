from src.core.settings import get_settings
from src.utils.video import encode_frame
from src.utils.timestamps import get_utc_time
import requests
from typing import Any
from src.utils.logger import log
from src.utils.video import decode_frame
from src.utils.video import crop_detections, draw_all_bounding_boxes
from src.utils.common import list_depth, combine_2_images
from PIL import Image
from src.utils.xai_modules.grad_explain import yolov8_heatmap
from src.utils.image import show_img


settings = get_settings()


def detect_car_thermal(frame, model, conf):
    """
    Detects objects in a thermal image using a pre-trained model.

    Args:
        frame (numpy.ndarray): The thermal image in which to detect objects.
        model (Model): The pre-trained model used for object detection in
        thermal images.
        conf (float): Confidence threshold for detection.

    Returns:
        list[dict]: A list of predictions containing detected objects, each
        with their bounding box coordinates, confidence scores and IDs.
    """
    detections = model.track(frame,
                             conf=conf,
                             verbose=False,
                             persist=True,
                             tracker="bytetrack.yaml")[0]
    return detections


def detect_people(frame, model, conf):
    """Detect people in a given frame using the specified model.

    This function utilizes the YOLO model to detect people in the input
    frame, filtering results based on the specified confidence threshold.

    Args:
        frame (Any): The input image frame in which to detect people.
        model (YOLO): The YOLO model used for detecting people.
        conf (float): The confidence threshold for filtering detections.

    Returns:
        Any: The detections produced by the model, which includes bounding
             boxes and confidence scores for detected people.
    """
    detections = model(
        frame,
        classes=settings.PEOPLE,
        conf=conf,
        verbose=False,
    )[0]
    return detections


def updated_car_status_thermal(car_id, car, detections, ids_detected,
                               cars_to_delete):
    """Update the status of a detected car based on thermal detections.

    This function checks the status of a car in the context of thermal
    detections, updating its attributes if a new detection is found,
    and tracking the number of frames it has been unseen. If a car is
    not detected for a specified number of frames, it marks the car for
    deletion.

    Args:
        car_id (int): The unique identifier for the car being updated.
        car (Car_Thermal): The car object that holds its current status.
        detections (dict): A dictionary containing detection information
                           for the current frame.
        ids_detected (List[int]): A list of car IDs detected in the current
                                   frame.
        cars_to_delete (List[int]): A list of car IDs to be marked for
        deletion.

    Returns:
        Tuple[List[int], List[int], bool]:
            - Updated list of car IDs to be deleted.
            - Updated list of detected car IDs.
            - A boolean indicating if the car status was updated.
    """
    updated = False
    car.frames_no_seen += 1
    if car_id in ids_detected:
        car.frames_no_seen = 0
        car_detected = detections.get(car_id, None)

        if car_detected["people"] is not None and car_detected[
                "people"] != car.people:
            car.car_coords = car_detected["car_coords"]
            car.car_confidence = car_detected["car_confidence"]
            car.people = car_detected["people"]
            car.people_coords = car_detected["people_coords"]
            car.people_confidence = car_detected["people_confidence"]
            updated = True

        ids_detected.remove(car_id)
    else:
        if car.frames_no_seen > 30:
            cars_to_delete.append(car_id)

    return cars_to_delete, ids_detected, updated


def send_car_update_thermal(frame, xai_image, car) -> requests.Response:
    """Send an update for a detected thermal car.

    This function constructs a payload containing information about the
    detected car, including its frame, coordinates, number of people,
    and confidence scores, and sends this information to the specified
    endpoint.

    Args:
        frame (Any): The image frame containing the detected car.
        car (Car_Thermal): The car object containing its current status
                           and detected attributes.

    Returns:
        requests.Response: The response object from the POST request,
                           which contains the server's response to the
                           update.
    """
    payload = {
        "frame": encode_frame(frame),
        "xai_frame": encode_frame(xai_image),
        "car_coordinates": car.car_coords,
        "people": car.people,
        "people_coordinates": car.people_coords,
        "people_confidence": car.people_confidence,
        "detected_at": get_utc_time().isoformat(),
    }
    response = requests.post(settings.people_url, json=payload)
    return response

def process_response(component: str,
                     response: dict,
                     visualize_detection: bool = False,
                     visualize_xai: bool = False) -> None:

    log.info("\n"
             "-------------------------------------\n"
             f"Component:  {component.title()}\n"
             f"Detected:   {response[component]}\n"
             f"Confidence: {response[f'{component}_confidence']}\n"
             f"DateTime:   {response['detected_at']}\n"
             "-------------------------------------\n")

    if visualize_detection:
        image = decode_frame(response["frame"])

        if list_depth(response[f"{component}_coordinates"]) == 1:
            bb_image = draw_all_bounding_boxes(
                image, [response[f"{component}_coordinates"]])
        elif list_depth(response[f"{component}_coordinates"]) == 2:
            bb_image = draw_all_bounding_boxes(
                image, response[f"{component}_coordinates"])
        show_img(bb_image)
    
    if visualize_xai:
        image = decode_frame(response["xai_frame"])
        show_img(image)

def xai_people_count(car_image, 
                     model_dir,
                     reference_model = "src/ml_models/yolov8n.pt"):
       
    model = yolov8_heatmap(
        weight=model_dir, 
        reference_model = reference_model,
        device = "cpu", 
        method = "EigenGradCAM", 
        layer=[10, 12, 14, 16, 18, -3],
        ratio=0.02,
        show_box=False,
        renormalize=False,
)
    explained_image = model(img=car_image)

    xai_image = combine_2_images(title1 = "Original Image",
                                 title2 = "Pixels classified as HUMAN",
                                 image1=car_image,
                                 image2=explained_image
                                 )
    return xai_image