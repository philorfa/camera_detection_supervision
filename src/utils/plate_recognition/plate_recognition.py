from src.utils.plate_recognition.plate_utils import (detect_car, detect_plate,
                                                     google_vision_ocr)
from src.utils.video import convert_coords_to_original_image, crop_detections


def recognize_plates(image, car_model, plate_model, ocr_model):
    """
    Detects cars and their license plates in an image, performs OCR on the
    detected plates, and returns the relevant information.

    This function:
    1. Detects cars in the input image using the provided car detection model.
    2. Crops each detected car and runs a plate detection model to find
       license plates within the cropped car region.
    3. Ensures that only one plate is detected per car; raises an exception
       if multiple plates are detected for a single car.
    4. Converts the plate coordinates to the original image scale.
    5. Crops the detected plate and performs Optical Character Recognition
        (OCR) on the cropped plate using Google Cloud Vision.
    6. Returns a list of detected cars with their coordinates, plate
        coordinates, plate confidence, and recognized text from OCR.

    Args:
        image (numpy.ndarray): The input image in which to detect cars and
                                plates.
        car_model (YOLO): The pre-trained YOLO model used for car detection.
        plate_model (YOLO): The pre-trained YOLO model used for license plate
                            detection.
        ocr_model (vision.ImageAnnotatorClient): The OCR model for recognizing
                                                text on plates.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains
            information
        about a detected car and its associated license plate (if any). Each
        dictionary includes:
        - 'car_coords' (list[float]): Bounding box coordinates of the car.
        - 'car_confidence' (float): Confidence score for the detected car.
        - 'plate_coords' (list[float], optional): Bounding box coordinates of
            the plate
          in the original image scale (if a plate is detected).
        - 'plate_confidence' (float, optional): Confidence score for the
        detected plate.
        - 'text' (str, optional): Recognized text from the license plate (if
        OCR is successful).
        - 'text_conf' (float, optional): Confidence score for the recognized
        text.

    Raises:
        Exception: If more than one license plate is detected for a single car.
    """

    info = {}
    ids_detected = []
    car_detections = detect_car(image, car_model, conf=0.4)
    for car in car_detections.boxes.data.tolist():

        info[int(car[4])] = {
            "car_coords": car[:4],
            "car_confidence": car[5],
            "plate_coords": None,
            "plate_confidence": None,
            "text": None,
            "text_conf": None
        }

        ids_detected.append(int(car[4]))
        zoomed_in_car_rgb = crop_detections(image, car)

        plate_detections = detect_plate(zoomed_in_car_rgb, plate_model)

        if len(plate_detections) > 2:
            raise Exception("Found 2 plates for 1 car")

        elif len(plate_detections) == 1:

            plate = plate_detections.boxes.data.tolist()[0]
            old_pl_coords = convert_coords_to_original_image(
                car[:4], plate[:4])

            zoomed_in_plate = crop_detections(zoomed_in_car_rgb, plate)
            text, conf = google_vision_ocr(zoomed_in_plate, ocr_model)

            info[int(car[4])].update({
                "plate_coords": old_pl_coords,
                "plate_confidence": plate[4],
                "text": text,
                "text_conf": conf
            })

    return info, ids_detected
