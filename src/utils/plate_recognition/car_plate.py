class Car_RGB:
    # Class attribute to store all cars
    cars_dict = {}

    def __init__(self, car_id, car_coords, car_confidence, plate_coords,
                 plate_text, text_confidence, frames_no_seen):
        self.id = car_id

        # Coordinates of the car
        self.car_coords = car_coords

        # Confidence score for car detection
        self.car_confidence = car_confidence

        # Coordinates of the license plate
        self.plate_coords = plate_coords

        # License plate text
        self.license_plate = plate_text

        # Confidence score for license plate detection
        self.text_confidence = text_confidence

        # The last frame in which the car was seen
        self.frames_no_seen = frames_no_seen

        # Add the car to the dictionary using its id as the key
        Car_RGB.cars_dict[self.id] = self

    @classmethod
    def create_car(cls, car_id, car_info):

        car_coords = car_info["car_coords"]
        car_confidence = car_info["car_confidence"]
        plate_coords = car_info["plate_coords"]
        plate_text = car_info["text"]
        text_confidence = car_info["text_conf"]

        new_car = cls(car_id,
                      car_coords,
                      car_confidence,
                      plate_coords,
                      plate_text,
                      text_confidence,
                      frames_no_seen=0)
        return new_car

    def __repr__(self):
        return (f"Car(id={self.id}, \
                license_plate={self.license_plate}, "
                f"car_coords={self.car_coords}, \
                car_confidence={self.car_confidence}, "
                f"plate_coords={self.plate_coords}, \
                text_confidence={self.text_confidence}, "
                f"frames_no_seen={self.frames_no_seen})")

    @classmethod
    def print_all_cars(cls):
        # Check if there are no cars
        if not cls.cars_dict:
            print("No cars available.")
            return

        print("List of Cars:")
        for car_id, car in cls.cars_dict.items():
            print(car)

    @classmethod
    def get_car_by_id(cls, car_id):
        return cls.cars_dict.get(car_id, None)
