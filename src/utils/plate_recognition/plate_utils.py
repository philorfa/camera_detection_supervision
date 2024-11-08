import cv2
from google.cloud import vision
import re
import requests
from typing import List, Tuple, Any, Dict
from src.core.settings import get_settings
from src.utils.video import encode_frame
from src.utils.timestamps import get_utc_time
from src.utils.xai_modules.lrp_explain import YOLOv8LRP
from typing import List, Any
from src.utils.video import decode_frame, draw_all_bounding_boxes
from src.utils.image import show_img
from src.utils.video import crop_detections
from src.utils.logger import log
from src.utils.common import list_depth
import torchvision
from PIL import Image

settings = get_settings()


def detect_car(frame, model, conf):
    """
    Detects vehicles in a video frame using a given YOLO model.

    Args:
        frame (numpy.ndarray): The input frame in which to detect vehicles.
        model (YOLO): The YOLO object detection model used for vehicle
                    detection.
        conf (float): The confidence threshold for vehicle detection.

    Returns:
        YOLO.Detections: Detected vehicles in the frame with bounding box
        coordinates and confidence scores.
    """
    car_detections = model.track(frame,
                                 conf=conf,
                                 classes=settings.VEHICLES,
                                 verbose=False,
                                 persist=True,
                                 tracker="bytetrack.yaml")[0]

    return car_detections


def detect_plate(frame, model):
    """
    Detects license plates in a cropped image of a vehicle using a given YOLO
    model.

    Args:
        frame (numpy.ndarray): Cropped image of the vehicle where the license
        plate is expected to be.
        model (YOLO): The YOLO object detection model used for plate detection.

    Returns:
        YOLO.Detections: Detected license plates in the frame with bounding box
        coordinates and confidence scores.
    """
    plate_detections = model(frame, conf=0.4, verbose=False)[0]
    return plate_detections


def google_vision_ocr(image, client):
    """
    Performs Optical Character Recognition (OCR) on an image using Google
    Cloud Vision API.

    Args:
        image (numpy.ndarray): The input image of the license plate to perform
        OCR on.
        client (vision.ImageAnnotatorClient): The Google Cloud Vision API
        client for OCR.

    Returns:
        tuple:
            - text (str): The detected text from the license plate.
            - confidence (float): The confidence score for the detected text.

    Raises:
        GoogleAPIError: If the OCR fails or if no text is detected.
    """
    text_detection_params = vision.TextDetectionParams(
        enable_text_detection_confidence_score=True)
    image_context = vision.ImageContext(
        text_detection_params=text_detection_params)

    image = vision.Image(content=cv2.imencode(".jpg", image)[1].tostring())
    response = client.text_detection(image=image, image_context=image_context)
    full_text = response.full_text_annotation
    check = full_text.text + "check"
    if check == "check":
        return "Cannot read", 0.0

    # Clean up detected text by removing special characters
    text = re.sub(r"[^a-zA-Z0-9 \n\.]", " ", full_text.text)
    return text, full_text.pages[0].confidence


def updated_car_status_rgb(
        car_id: int, car: Any, detections: Dict[int, Dict[str, Any]],
        ids_detected: List[int],
        cars_to_delete: List[int]) -> Tuple[List[int], List[int], bool]:
    """Update the status of a detected car based on current detections.

    This function checks if the detected car is present in the current frame,
    updates its attributes if new information is available, and determines
    if it should be marked for deletion after not being detected for a set
    number of frames.

    Args:
        car_id (int): The ID of the car being updated.
        car (Any): The car object containing its current status.
        detections (Dict[int, Dict[str, Any]]): Current detections with their
        attributes.
        ids_detected (List[int]): List of detected car IDs in the current
        frame.
        cars_to_delete (List[int]): List of car IDs marked for deletion.

    Returns:
        Tuple[List[int], List[int], bool]:
            - Updated list of car IDs to delete.
            - Updated list of detected car IDs.
            - Boolean indicating if the car's status was updated.
    """

    updated = False
    car.frames_no_seen += 1
    if car_id in ids_detected:
        car.frames_no_seen = 0
        car_detected = detections.get(car_id, None)

        if car_detected[
                "text"] is not None and car.text_confidence < car_detected[
                    "text_conf"]:
            car.car_coords = car_detected["car_coords"]
            car.car_confidence = car_detected["car_confidence"]
            car.plate_coords = car_detected["plate_coords"]
            car.license_plate = car_detected["text"]
            car.text_confidence = car_detected["text_conf"]
            updated = True

        ids_detected.remove(car_id)
    else:
        if car.frames_no_seen > 30:
            cars_to_delete.append(car_id)

    return cars_to_delete, ids_detected, updated


def send_car_update_rgb(frame: Any, 
                        xai_image: Any,
                        car: Any) -> requests.Response:
    """Send an update about the detected car to the specified endpoint.

    This function encodes the current frame and sends a JSON payload
    containing the car's coordinates, license plate information, and
    the timestamp of detection to a predefined URL.

    Args:
        frame (Any): The current video frame containing the detected car.
        car (Any): The car object containing its properties.

    Returns:
        requests.Response: The response object from the POST request.
    """
    payload = {
        "frame": encode_frame(frame),
        "xai_frame": encode_frame(xai_image),
        "car_coordinates": car.car_coords,
        "plate_coordinates": car.plate_coords,
        "plate": car.license_plate,
        "plate_confidence": car.text_confidence,
        "detected_at": get_utc_time().isoformat(),
    }
    response = requests.post(settings.plate_url, json=payload)
    return response

def process_response(component: str,
                     response: dict,
                     visualize_detection: bool = False,
                     visualize_xai: bool = False) -> None:
    """Process the response from a detection component.

    Logs the details of the detection and optionally visualizes the results by
    drawing bounding boxes on the detected objects.

    Args:
        component (str): The name of the detection component (e.g., 'plate').
        response (Dict[str, Union[str, float, List]): A dictionary containing
        detection results, including:
            - component: Detected object.
            - component_confidence: Confidence score of the detection.
            - detected_at: Timestamp of the detection.
            - frame: The frame containing the detected objects.
            - component_coordinates: Coordinates of the detected objects.
        visualize (bool, optional): If True, visualize the detections on the
        image.
            Defaults to False.

    Returns:
        None
    """

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

def xai_plate_recognition(plate_image, 
                          plate_model):
       
    # Convert the NumPy array to a PIL Image
    plate_image = Image.fromarray(plate_image)
    desired_size = (640, 640)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(desired_size),
        torchvision.transforms.ToTensor()
                 ])
    plate_frame = transform(plate_image)
    
    lrp_plate = YOLOv8LRP(plate_model, power=2, eps=1, device='cpu')

    explanation_lrp_plate = lrp_plate.explain(plate_frame, cls="license_plate", contrastive=False).cpu()
    
    xai_image = lrp_plate.plot_explanation(frame=plate_frame, explanation = explanation_lrp_plate, contrastive=True, cmap='seismic', 
                               title='Pixels that detector classified as license plate')
    
    return xai_image