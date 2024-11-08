from src.core.settings import get_settings
from typing import List, Tuple, Any, Dict
import requests
from src.utils.timestamps import get_utc_time
from src.utils.video import encode_frame
from src.utils.logger import log
from src.utils.video import decode_frame
from src.utils.image import show_images_grid
from openai import OpenAI
from textwrap import dedent
from typing import Literal
from pydantic import BaseModel

settings = get_settings()

def detect_car(frame: Any, model: Any, conf: float) -> Any:
    """
    Detects vehicles in a video frame using a given YOLO model.

    Args:
        frame (numpy.ndarray): The input frame in which to detect vehicles.
        model (YOLO): The YOLO object detection model used for vehicle detection.
        conf (float): The confidence threshold for vehicle detection.

    Returns:
        YOLO.Detections: Detected vehicles in the frame with bounding box coordinates and confidence scores.
    """

    car_detections = model.track(
        frame,
        conf=conf,
        classes=settings.VEHICLES,
        verbose=False,
        persist=True,
        tracker="bytetrack.yaml"
    )
    return car_detections[0]

def updated_car_images_rgb(
    car_id: int, car: Any, detections: Dict[int, Dict[str, Any]], 
    ids_detected: List[int], cars_to_delete: List[int]
) -> Tuple[List[int], List[int], bool]:
    """
    Updates detected car images in RGB, preparing for identification if conditions are met.

    Args:
        car_id (int): Unique identifier of the car.
        car (Any): Car object containing details and lists of images.
        detections (dict): Dictionary of car detections with bounding box coordinates and confidence scores.
        ids_detected (list): List of detected car IDs in the current frame.
        cars_to_delete (list): List to accumulate IDs of cars to remove.

    Returns:
        Tuple[List[int], List[int], bool]: Updated `cars_to_delete`, `ids_detected`, 
        and a boolean indicating if the car is ready for identification.
    """
    show_at_frame_num = 20
    ready_for_identification = False

    car.frames_no_seen += 1

    if car_id in ids_detected:
        car.frames_no_seen = 0
        if car.frames_detected < show_at_frame_num:
            car_detected = detections.get(car_id)
            car.list_original.append(car_detected["original_image"])
            car.list_cropped.append(car_detected["cropped_image"])
            car.frames_detected += 1

        if car.frames_detected == show_at_frame_num:
            ready_for_identification = True
            car.frames_detected += 1

        ids_detected.remove(car_id)
    else:
        if car.frames_no_seen > 30:
            cars_to_delete.append(car_id)

    return cars_to_delete, ids_detected, ready_for_identification

class Properties(BaseModel):
    """
    Model representing car properties with constraints on color and type, including confidence scores and reasoning.

    Attributes:
        color (Literal): The color of the car, constrained to specific values.
        color_confidence (float): Confidence score for the identified color.
        color_reasoning (str): Explanation of the reasoning behind the color choice.
        type (Literal): The type of the car, constrained to specific values.
        type_confidence (float): Confidence score for the identified type.
        type_reasoning (str): Explanation of the reasoning behind the type choice.
    """
    color: Literal["Red", "Blue", "Black", "White", "Gray", "Silver", "Green", "Yellow", "Orange", "Brown"]
    color_confidence: float
    color_reasoning: str
    type: Literal["Sedan", "SUV", "Coupe", "Hatchback", "Truck", "Convertible"]
    type_confidence: float
    type_reasoning: str

def identify_properties(car: Any) -> Any:
    """
    Identifies car properties such as color and type, along with confidence scores and reasoning, using a language model.

    Args:
        car (Any): Car object with cropped images for analysis.

    Returns:
        Any: The car object with updated color, type, confidence scores, and reasoning.
    """
    log.info("== Identifying car color and type ==")
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # System prompt to instruct the model on its task, confidence score, and reasoning requirement.
    system_role = dedent('''
    You are an advanced AI model trained to assist in identifying car characteristics based on visual data.
    You will receive an array of cropped images, all referring to the same car.
    Your task is to identify two key characteristics for this car: (1) its color and (2) its type.

    For each characteristic, please provide:
    - A "confidence score" — a numerical value between 0 and 1 — that represents your level of certainty in each prediction.
      - A score of 1 means you are completely certain of the prediction.
      - A score closer to 0 indicates low confidence in the prediction.
    - A detailed text explanation for each choice, with a focus on the car type:
        - Identify key visual features or contextual details that led you to choose this type.
        - Clearly explain why you did not choose each of the other possible types, listing specific characteristics that differentiate the chosen type from the others.

    Use the following criteria for each car type when making your decision:

    - **Sedan**: 
        1. Low ground clearance, typically close to the ground.
        2. Four doors with a distinct, separated trunk compartment.
        3. Moderate overall size, generally neither too large nor compact.
        
    - **SUV**: 
        1. High ground clearance, suitable for off-road or rugged terrain.
        2. Boxy and tall body shape with a large cabin space.
        3. Often equipped with roof rails and larger wheels for rugged use.

    - **Coupe**: 
        1. Two doors and a sportier, streamlined profile.
        2. Sloping roofline, giving a sleeker, aerodynamic appearance.
        3. Compact or medium size, often lower to the ground than sedans or SUVs.
        
    - **Hatchback**: 
        1. Compact body with a rear hatch door that opens upward.
        2. Two or four doors with a continuous cabin space (no separated trunk).
        3. Generally small and efficient, with a short rear overhang.

    - **Truck**: 
        1. Distinct cargo bed separated from the cabin.
        2. High ground clearance, often with larger tires for utility use.
        3. Typically larger in size, with a bulky, robust frame for carrying loads.

    - **Convertible**: 
        1. Retractable roof or soft top, often exposed when down.
        2. Sporty, low-profile body, similar to coupes in structure.
        3. Typically has two doors, with a compact and stylish design.

    For color, select from a predefined list of 10 possible colors.
    For car type, choose from a predefined list of 6 types.
''')

    # User prompt to request specific information from the model.
    user_role = dedent('''
    Based on the provided images, please identify the following characteristics of the car:
    
    Car Color: Choose from the following colors - Red, Blue, Black, White, Gray, Silver, Green, Yellow, Orange, Brown
    Car Type: Choose from the following types - Sedan, SUV, Coupe, Hatchback, Truck, Convertible

    For each characteristic, please provide:
    - The identified car color and type.
    - A confidence score between 0 and 1 that reflects your certainty for each choice.
    - An explanation of the visual or contextual reasons that led you to make each choice.

    For the car type, please consider visual criteria specific to each type, such as body shape, ground clearance, and size. Explain the key points that led you to your chosen type and why you ruled out other types based on their distinct characteristics.

    Example:
        Color: Black
        Color Confidence: 0.85
        Color Reasoning: The images suggest a dark tone with minimal reflection, consistent with black paint.
        
        Type: SUV
        Type Confidence: 0.90
        Type Reasoning: The car appears large with a higher ground clearance, typical of SUVs. The shape of the body and roofline further support this classification.
            - Did not choose Sedan: The car is higher off the ground and bulkier than typical sedans.
            - Did not choose Coupe: Coupes are generally smaller with a sportier design.
            - Did not choose Truck: The car does not have a separate cargo area or the rugged build typical of trucks.
            - Did not choose Convertible: There is no indication of a retractable roof, which is characteristic of convertibles.
            - Did not choose Hatchback: The car lacks the compact shape and rear-door style of hatchbacks.
''')

    # Prepare the input content, including images.
    input_content = [{"type": "text", "text": user_role}]
    for i, image in enumerate(car.list_cropped):
        if i % 5 == 0:  # Encode every 5th image to avoid excessive inputs.
            base64_image = encode_frame(image)
            input_content.append({"type": "image_url", 
                                  "image_url": {
                                      "url": f"data:image/jpeg;base64,{base64_image}",
                                      "detail": "high"
                                      }})

    # Construct the messages for the completion call.
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_role}]},
        {"role": "user", "content": input_content}
    ]

    # Request completion from the model with the Properties response format.
    completion = client.beta.chat.completions.parse(
        model=settings.OPENAI_MODEL,
        messages=messages,
        response_format=Properties,
    )
    message = completion.choices[0].message

    # Populate the car object with the results.
    car.car_color = message.parsed.color
    car.color_confidence = message.parsed.color_confidence
    car.color_reasoning = message.parsed.color_reasoning
    car.car_type = message.parsed.type
    car.type_confidence = message.parsed.type_confidence
    car.type_reasoning = message.parsed.type_reasoning

    return car

def send_car_properties_rgb(car: Any) -> requests.Response:
    """
    Sends car properties and images to an external API.

    Args:
        car (Any): Car object containing images and identified properties.

    Returns:
        requests.Response: Response object from the API after submitting the car data.
    """
    encoded_original_frames = [encode_frame(frame) for frame in car.list_original]
    encoded_cropped_frames = [encode_frame(frame) for frame in car.list_cropped]

    payload = {
        "original_frames": encoded_original_frames,
        "cropped_frames": encoded_cropped_frames,
        "car_color": car.car_color,
        "color_confidence": car.color_confidence,
        "color_reasoning": car.color_reasoning,
        "car_type": car.car_type,
        "type_confidence": car.type_confidence,
        "type_reasoning": car.type_reasoning,
        "detected_at": get_utc_time().isoformat(),
    }

    response = requests.post(settings.properties_url, json=payload)
    return response

def process_response_modifications(
    component: str, response: Dict[str, Any], visualize: bool = False
) -> None:
    """
    Logs detected modifications and optionally visualizes the car images.

    Args:
        component (str): The component responsible for detecting modifications.
        response (dict): The response containing detected car details.
        visualize (bool): Flag to indicate whether to display images.

    Returns:
        None
    """
    log.info("\n"
             "-------------------------------------\n"
             f"Component:  {component.title()}\n"
             f"Detected_Color:   {response['car_color']}\n"
             f"Color Confidence Score: {response['color_confidence']}\n"
             f"Color Reasoning: {response['color_reasoning']}\n"
             f"Detected_Type: {response['car_type']}\n"
             f"Type Confidence Score: {response['type_confidence']}\n"
             f"Type Reasoning: {response['type_reasoning']}\n"
             f"DateTime:   {response['detected_at']}\n"
             "-------------------------------------\n")

    if visualize:
        original_images = [decode_frame(frame) for frame in response["original_frames"]]
        cropped_images = [decode_frame(frame) for frame in response["cropped_frames"]]

        show_images_grid(original_images)
        show_images_grid(cropped_images)
