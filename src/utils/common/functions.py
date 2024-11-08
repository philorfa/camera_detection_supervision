from typing import Union, List, Any
from src.utils.video import decode_frame, draw_all_bounding_boxes
from src.utils.image import show_img
from src.utils.logger import log
import cv2
import numpy as np


def list_depth(nested_list: Union[List, Any]) -> int:
    """Calculate the depth of a nested list.

    Args:
        nested_list (Union[List, Any]): The nested list whose depth is to be
        calculated.

    Returns:
        int: The depth of the nested list.
    """
    if isinstance(nested_list, list):
        return 1 + max(list_depth(item)
                       for item in nested_list) if nested_list else 1
    else:
        return 0
    
def resize_with_aspect_ratio(image, desired_width, desired_height):
    """Resize an image while maintaining its aspect ratio.

    Args:
        image (np.ndarray): The image to resize.
        desired_width (int): The desired width.
        desired_height (int): The desired height.

    Returns:
        np.ndarray: The resized image with padding to match desired dimensions.
    """
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height

    # Determine scaling factor and new dimensions
    if desired_width / desired_height > aspect_ratio:
        # Height is the constraining dimension
        new_height = desired_height
        new_width = int(desired_height * aspect_ratio)
    else:
        # Width is the constraining dimension
        new_width = desired_width
        new_height = int(desired_width / aspect_ratio)

    # Resize the image
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new image of desired size with white background
    image_padded = np.ones((desired_height, desired_width, 3), dtype=np.uint8) * 255  # White background

    # Compute top-left corner for centering the image
    x_offset = (desired_width - new_width) // 2
    y_offset = (desired_height - new_height) // 2

    # Place the resized image onto the padded image
    image_padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = image_resized

    return image_padded
    
def combine_2_images(title1, title2, image1, image2):
    # Desired dimensions for resizing
    desired_height = 400
    desired_width = 500

    # Resize images while maintaining aspect ratio
    img1_resized = resize_with_aspect_ratio(image1, desired_width, desired_height)
    img2_resized = resize_with_aspect_ratio(image2, desired_width, desired_height)

    # Font settings for text rendering
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 0)  # Black text

    # Determine text size to create space for titles
    (text_width1, text_height1), base_line1 = cv2.getTextSize(title1, font, font_scale, font_thickness)
    (text_width2, text_height2), base_line2 = cv2.getTextSize(title2, font, font_scale, font_thickness)
    title_height = max(text_height1, text_height2) + max(base_line1, base_line2) + 20  # Extra padding

    # Since images are resized to desired dimensions, use those for combined image
    height, width = desired_height, desired_width

    # Create a new image with space for titles and both images side by side
    combined_height = height + title_height
    combined_width = width * 2
    combined_image = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255  # White background

    # Place the first image
    combined_image[title_height:title_height+height, 0:width] = img1_resized

    # Place the second image
    combined_image[title_height:title_height+height, width:width*2] = img2_resized

    # Center positions for titles
    text_x1 = int((width - text_width1) / 2)
    text_y1 = int((title_height + text_height1) / 2) - 10  # Adjust for visual centering
    text_x2 = width + int((width - text_width2) / 2)
    text_y2 = text_y1  # Same as first title

    # Add titles to the image
    cv2.putText(combined_image, title1, (text_x1, text_y1), font, font_scale, text_color, font_thickness)
    cv2.putText(combined_image, title2, (text_x2, text_y2), font, font_scale, text_color, font_thickness)
    return combined_image


def process_response(component: str,
                     response: dict,
                     visualize: bool = False) -> None:
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

    if visualize:
        image = decode_frame(response["frame"])

        if list_depth(response[f"{component}_coordinates"]) == 1:
            bb_image = draw_all_bounding_boxes(
                image, [response[f"{component}_coordinates"]])
        elif list_depth(response[f"{component}_coordinates"]) == 2:
            bb_image = draw_all_bounding_boxes(
                image, response[f"{component}_coordinates"])
        show_img(bb_image)
    