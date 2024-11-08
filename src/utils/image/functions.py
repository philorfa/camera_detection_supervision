import cv2
import numpy as np


def show_img(image: np.ndarray) -> None:
    """Display an image in a window.

    This function uses OpenCV to display the provided image in a window.
    The window will wait for a key press before closing.

    Args:
        image (np.ndarray): The image to be displayed.

    Returns:
        None
    """
    cv2.imshow("image", image)
    cv2.waitKey(0)
    # Add this line to close the window after a key is pressed.
    cv2.destroyAllWindows()

def show_images_grid(images,title="Image Grid"):
        # Determine grid size
        grid_rows = int(np.ceil(np.sqrt(len(images))))
        grid_cols = int(np.ceil(len(images) / grid_rows))
        
        # Find the maximum width and height of the images for a uniform grid
        max_height = max(image.shape[0] for image in images)
        max_width = max(image.shape[1] for image in images)
        
        # Create a blank canvas for the grid
        grid_image = np.zeros((grid_rows * max_height, grid_cols * max_width, 3), dtype=np.uint8)
        
        # Place images in the grid
        for idx, img in enumerate(images):
            row = idx // grid_cols
            col = idx % grid_cols
            start_y = row * max_height
            start_x = col * max_width
            grid_image[start_y:start_y + img.shape[0], start_x:start_x + img.shape[1]] = img
        
        # Display the grid image
        cv2.imshow(title, grid_image)
        cv2.waitKey(0)  # Wait for any key press
        cv2.destroyAllWindows()