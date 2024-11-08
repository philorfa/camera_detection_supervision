from src.utils.thermal_counting.thermal_utils import (detect_car_thermal,
                                                      detect_people)
from src.utils.video import convert_coords_to_original_image, crop_detections


def thremal_count(image, car_model, people_model, confidence_thresh):
    """Detect cars and people in a thermal image.

    This function uses the provided car and people detection models to
    analyze the given thermal image, identifying detected cars and
    counting the number of people within each car's detected area.

    Args:
        image (Any): The thermal image to analyze.
        car_model (YOLO): The YOLO model for car detection.
        people_model (YOLO): The YOLO model for people detection.
        confidence_thresh (float): The confidence threshold for people
        detection.

    Returns:
        Tuple[Dict[int, Dict[str, Any]], List[int]]:
            - A dictionary containing detection information for each car,
              indexed by car ID.
            - A list of IDs for the detected cars.
    """

    info = {}
    ids_detected = []
    car_detections = detect_car_thermal(image, car_model, conf=0.3)
    for car in car_detections.boxes.data.tolist():

        info[int(car[4])] = {
            "car_coords": car[:4],
            "car_confidence": car[5],
            "people": None,
            "people_coords": None,
            "people_confidence": None
        }
        ids_detected.append(int(car[4]))
        zoomed_in_car_thermal = crop_detections(image, car)

        people_detection = detect_people(zoomed_in_car_thermal,
                                         people_model,
                                         conf=confidence_thresh)

        human_counting = 0
        human_confidence = []
        people_coordinates = []
        for human in people_detection.boxes.data.tolist():

            human_counting = +1
            human_confidence.append(human[4])

            person_coordinates = convert_coords_to_original_image(
                car[:4], human[:4])

            people_coordinates.append(person_coordinates)

        if human_counting > 0:
            info[int(car[4])].update({
                "people": human_counting,
                "people_coords": people_coordinates,
                "people_confidence": human_confidence
            })

    return info, ids_detected
