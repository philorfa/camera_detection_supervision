import cv2
import numpy as np
from src.utils.logger import log

class Car_Properties:
    # Class attribute to store all cars
    cars_dict = {}

    def __init__(self, car_id, 
                 car_color,
                 color_confidence, 
                 color_reasoning,
                 car_type, 
                 type_confidence,
                 type_reasoning,
                 frames_detected, 
                 list_cropped, 
                 list_original, 
                 frames_no_seen):
        
        self.id = car_id

        # Color of car
        self.car_color = car_color

        # Confidence Score for Color
        self.color_confidence = color_confidence

        # Color Reasoning
        self.color_reasoning = color_reasoning

        # Type of car
        self.car_type = car_type

        # Confidence Score for Type
        self.type_confidence = type_confidence

        # Type Reasoning
        self.type_reasoning = type_reasoning

        # Number of frames that the vehicle was detected
        self.frames_detected = frames_detected

        # List of cropped images
        self.list_cropped = list_cropped

        # List of original images
        self.list_original = list_original

        # The last frame in which the car was seen
        self.frames_no_seen = frames_no_seen

        # Add the car to the dictionary using its id as the key
        Car_Properties.cars_dict[self.id] = self

    @classmethod
    def create_car(cls, car_id, car_info):

        image_cropped = car_info["cropped_image"]
        image_original = car_info["original_image"]

        new_car = cls(car_id,
                      car_color=None,
                      color_confidence=None,
                      color_reasoning=None,
                      car_type=None,
                      type_confidence=None,
                      type_reasoning=None,
                      list_cropped=[image_cropped],
                      list_original=[image_original],
                      frames_detected=1,
                      frames_no_seen=0)
        return new_car

    def __repr__(self):
        return (f"Car(id={self.id}",
                f"Car Color={self.car_color}",
                f"Color Confidence={self.color_confidence}",
                f"Color Reasoning={self.color_reasoning}",
                f"Car Type={self.car_type}",
                f"Type Confidence={self.type_confidence}",
                f"Type Reasoning={self.type_reasoning}",
                f"Frames_detected={self.frames_detected}",
                f"Length of original list={len(self.list_original)}",
                f"Length of cropped list={len(self.list_original)}",
                f"Num of Frames since last seen={self.frames_no_seen}")

    @classmethod
    def print_all_cars(cls):
        # Check if there are no cars
        if not cls.cars_dict:
            log.info("No cars available.")
            
        else:
            print("List of Cars:")
            for car_id, car in cls.cars_dict.items():
                log.info(car.__repr__())
        
        return

    @classmethod
    def get_car_by_id(cls, car_id):
        return cls.cars_dict.get(car_id, None)
    
    def show_images_grid(self, images, title):
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

    def show_cropped_images(self):
        if self.list_cropped:
            self.show_images_grid(self.list_cropped, f"Cropped Images for Car ID {self.id}")
        else:
            log.error(f"No cropped images available for Car ID {self.id}")

    def show_original_images(self):
        if self.list_original:
            self.show_images_grid(self.list_original, f"Original Images for Car ID {self.id}")
        else:
            log.error(f"No original images available for Car ID {self.id}")