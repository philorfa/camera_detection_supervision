import cv2
import numpy as np
import base64
from src.core.settings import get_settings

settings = get_settings()


def crop_detections(frame, coordinates, scale=settings.SCALE):
    """Crop and resize the specified region of the frame.

    This function crops the area defined by the bounding box coordinates from
    the input frame and resizes it according to the specified scale.

    Args:
        frame (np.ndarray): The input image frame from which to crop.
        coordinates (List[float]): A list of bounding box coordinates in the
        format [x1, y1, x2, y2].
        scale (float, optional): The scaling factor for resizing the cropped
        region. Defaults to settings.SCALE.

    Returns:
        np.ndarray: The cropped and resized image region.
    """
    x1, y1, x2, y2 = map(int, coordinates[:4])
    region = frame[y1:y2, x1:x2]

    zoomed_in_region = cv2.resize(region,
                                  None,
                                  fx=scale,
                                  fy=scale,
                                  interpolation=cv2.INTER_LANCZOS4)

    return zoomed_in_region


def check_resolution(video_path):
    """Check the resolution of the video file.

    This function retrieves the height and width of the video specified by the
    path.

    Args:
        video_path (str): The path to the video file.

    Returns:
        Tuple[float, float]: A tuple containing the height and width of the
        video.
    """

    vid = cv2.VideoCapture(video_path)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    return height, width


def convert_coords_to_original_image(bbox, coords, scale=settings.SCALE):
    """Convert new coordinates to original image coordinates based on scaling.

    This function takes bounding box coordinates and scaled coordinates,
    converting the scaled coordinates back to the original image's coordinate
    system.

    Args:
        bbox (List[float]): The original bounding box coordinates
        [x1, y1, x2, y2].
        coords (List[float]): The new bounding box coordinates after scaling.
        scale (float, optional): The scaling factor used. Defaults to
        settings.SCALE.

    Returns:
        Tuple[int, int, int, int]: The original image coordinates
        [old_x1, old_y1, old_x2, old_y2].
    """
    x1, y1, _, _ = bbox
    new_x1, new_y1, new_x2, new_y2 = coords

    # Scale down the new coordinates by a factor of 2
    old_x1 = x1 + (new_x1 // scale)
    old_y1 = y1 + (new_y1 // scale)
    old_x2 = x1 + (new_x2 // scale)
    old_y2 = y1 + (new_y2 // scale)

    return old_x1, old_y1, old_x2, old_y2


def draw_all_bounding_boxes(image, list_coordinates):
    """Draw bounding boxes on the image.

    This function takes an image and a list of bounding box coordinates and
    draws rectangles on the image for each bounding box.

    Args:
        image (np.ndarray): The image on which to draw the bounding boxes.
        list_coordinates (List[List[float]]): A list of bounding box
        coordinates.
            Each bounding box is defined by four values [x1, y1, x2, y2].

    Returns:
        np.ndarray: The image with the drawn bounding boxes.
    """
    for box in list_coordinates:
        if box:
            x1, y1, x2, y2 = map(int, box[:4])

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image


def return_frame_from_video(video_capture, frame_num):
    """Retrieve a specific frame from a video capture object.

    This function sets the position of the video capture to a specified
    frame number and retrieves that frame.

    Args:
        video_capture (cv2.VideoCapture): The video capture object from which
                                           to read the frame.
        frame_num (int): The frame number to retrieve from the video.

    Returns:
        Tuple[bool, Any]: A tuple where the first element is a boolean
                          indicating if the frame was successfully read,
                          and the second element is the retrieved frame
                          (or None if not successful).
    """

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    # return ret, frame
    return video_capture.read()


def encode_frame(frame: np.ndarray) -> str:
    """Encode an image frame to a base64-encoded string.

    This function takes a NumPy array representing an image and encodes it
    as a JPEG image. The encoded image is then converted to a base64 string
    for easier transmission or storage.

    Args:
        frame (np.ndarray): A NumPy array representing the image to be encoded.

    Returns:
        str: A base64-encoded string representation of the encoded image.
    """
    _, buffer = cv2.imencode(".jpg", frame)
    # Convert to base64
    frame_base64 = base64.b64encode(buffer).decode("utf-8")
    return frame_base64


def decode_frame(frame: str) -> np.ndarray:
    """Decode a base64-encoded image frame.

    This function takes a base64-encoded string representation of an image,
    decodes it, and converts it into a NumPy array representing the image
    in color.

    Args:
        frame (str): A base64-encoded string representing the image frame.

    Returns:
        np.ndarray: A NumPy array representing the decoded image.
    """
    frame_data = base64.b64decode(frame)
    nparr = np.frombuffer(frame_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image
