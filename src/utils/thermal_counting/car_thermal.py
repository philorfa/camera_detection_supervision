class Car_Thermal:
    # Class attribute to store all cars
    cars_dict = {}

    def __init__(self, car_id, car_coords, car_confidence, people,
                 people_coords, people_confidence, frames_no_seen):
        self.id = car_id

        # Coordinates of the car
        self.car_coords = car_coords

        # Confidence score for car detection
        self.car_confidence = car_confidence

        # Number of people inside the car
        self.people = people

        # Coordinates of people
        self.people_coords = people_coords

        # Confidence score for people
        self.people_confidence = people_confidence

        # The last frame in which the car was seen

        self.frames_no_seen = frames_no_seen

        # Add the car to the dictionary using its id as the key
        Car_Thermal.cars_dict[self.id] = self

    @classmethod
    def create_car(cls, car_id, car_info):

        car_coords = car_info["car_coords"]
        car_confidence = car_info["car_confidence"]
        people = car_info["people"]
        people_coords = car_info["people_coords"]
        people_confidence = car_info["people_confidence"]

        new_car = cls(car_id,
                      car_coords,
                      car_confidence,
                      people,
                      people_coords,
                      people_confidence,
                      frames_no_seen=0)
        return new_car

    def __repr__(self):
        return (f"Car(id={self.id}, \
                Number of People={self.people}, "
                f"car_coords={self.car_coords}, \
                car_confidence={self.car_confidence}, "
                f"poeple_coords={self.people_coords_coords}, \
                people_confidence={self.people_confidence}, "
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
