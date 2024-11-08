import time
import os
import cv2
from tqdm import tqdm
import numpy as np
from roboflow import Roboflow
import yaml

from util import (detect_in_thermal,
                  crop_detections_thermal,
                  update_results_thermal,
                  create_frame_visual_thermal,
                  convert_coords_to_original_image)

try:
    from sort.sort import Sort
except (Exception, ):
    raise Exception('You need sort module!')

VID_FOLDER = 6
WRITE_VID_THERMAL = True
MAX_PEOPLE = False

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

roboflow_api_key = config['roboflow_api']

if __name__ == '__main__':
    print("> Starting Initialization ...")
    init_start_time = time.time()

    # report folder
    report_folder = os.path.join('./reports_thermal', str(VID_FOLDER))
    os.makedirs(report_folder, exist_ok=True)

    # load object detection models
    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace("thermal-dwi24").project("thermal-car")
    car_model = project.version(1).model

    # people counting model
    project = rf.workspace("thermal-dwi24").project("thermal-tdwzh")
    people_model = project.version(1).model

    # load tracker
    car_tracker = Sort()

    # load video
    thermal_video = 'thermal.mp4'
    thermal_video_path = os.path.join('./thermal_data', str(VID_FOLDER),
                                      thermal_video)

    cap_thermal = cv2.VideoCapture(thermal_video_path)

    # output csv
    report_csv = os.path.join(report_folder, 'output_report.csv')

    # live report csv
    if MAX_PEOPLE:
        live_report_csv_path = os.path.join(report_folder,
                                            "report_max_people.csv")
    else:
        live_report_csv_path = os.path.join(report_folder, "report_live.csv")

    with open(live_report_csv_path, 'w') as f:
        f.write('{},{},{},{},{}\n'.format('frame_nmr', 'sec_in_vid(sec)',
                                          'frame_time_processed(sec)',
                                          'car_id', 'people_in_car'))
        f.close()

    with open(report_csv, 'w') as f:
        f.write('{},{},{},{}\n'.format('car_id',
                                       'first_frame',
                                       'last_frame',
                                       'people'))
        f.close()

    # output live rgb video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps_thermal = cap_thermal.get(cv2.CAP_PROP_FPS)
    width_thermal = int(cap_thermal.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_thermal = int(cap_thermal.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if WRITE_VID_THERMAL:
        if MAX_PEOPLE:
            output_live_vid = os.path.join(report_folder,
                                           'thermal_max_people.mp4')
        else:
            output_live_vid = os.path.join(report_folder,
                                           'thermal_live.mp4')

        out_thermal = cv2.VideoWriter(output_live_vid, fourcc, fps_thermal,
                                      (width_thermal, height_thermal))

        # check video resolution
        print("--- Thermal Video info ---")
        print("Resolution width x height: ", int(width_thermal), " x ",
              int(height_thermal))
        print("FPS ", fps_thermal)

        length_thermal = int(cap_thermal.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total Frames: ", length_thermal)

    results = {}
    max_people_car = []
    car_ids_frames = []

    # read frames
    frame_nmr = -1
    ret_thermal = True

    frame_count = length_thermal

    init_end_time = time.time() - init_start_time
    print("> Initialization time(sec) :", init_end_time)
    pbar = tqdm(total=frame_count)

    while ret_thermal and frame_nmr < frame_count:

        frame_start_time = time.time()

        frame_nmr += 1
        ret_thermal, frame_thermal = cap_thermal.read()
        pbar.update(1)

        if ret_thermal and frame_nmr < frame_count:

            results[frame_nmr] = {}
            car_info = []

            tracker_detections = []
            car_detections = detect_in_thermal(frame_thermal,
                                               car_model,
                                               conf=0.3)
            car_detected = False
            for car in car_detections:
                car_detected = True
                x = car['x']
                y = car['y']
                height = car['height']
                width = car['width']
                coordinates = []

                # x1
                coordinates.append(int(x - width / 2))
                # y1
                coordinates.append(int(y - height / 2))
                # x2
                coordinates.append(int(x + width / 2))
                # y2
                coordinates.append(int(y + height / 2))
                tracker_detections.append(coordinates)

                zoomed_in_car_thermal = crop_detections_thermal(
                    frame_thermal, coordinates)

                people_detection = detect_in_thermal(zoomed_in_car_thermal,
                                                     people_model,
                                                     conf=0.3)

                human_counting = 0
                human_confidence = []
                people_coordinates = []
                for preds in people_detection:
                    if (preds["class"] == 'human' and
                            preds["confidence"] > 0.65):
                        human_counting = +1
                        human_confidence.append(preds['confidence'])
                        x = preds['x']
                        y = preds['y']
                        height = preds['height']
                        width = preds['width']
                        person_coordinates = []

                        # x1
                        person_coordinates.append(int(x - width / 2))
                        # y1
                        person_coordinates.append(int(y - height / 2))
                        # x2
                        person_coordinates.append(int(x + width / 2))
                        # y2
                        person_coordinates.append(int(y + height / 2))

                        person_coordinates = convert_coords_to_original_image(
                            coordinates, person_coordinates)

                        people_coordinates.append(person_coordinates)

                if human_counting > 0:

                    people_car = {
                        'coords': coordinates,
                        'people': human_counting,
                        'conf': human_confidence,
                        'people_coords': people_coordinates
                    }

                else:
                    people_car = {
                        'coords': coordinates,
                        'people': 0,
                        'conf': "-",
                        'people_coords': "-"
                    }

                car_info.append(people_car)
            if car_detected:
                track_ids = car_tracker.update(np.asarray(tracker_detections))

                for track in track_ids:
                    updated = False
                    for entry in car_ids_frames:
                        if entry['car_id'] == track[4]:
                            entry['last_seen_frame'] = frame_nmr
                            updated = True

                    if not updated:
                        new_entry = {'car_id': track[4],
                                     'first_seen_frame': frame_nmr,
                                     'last_seen_frame': frame_nmr,
                                     'people': 0}
                        car_ids_frames.append(new_entry)

            results, max_people_car = update_results_thermal(
                results, car_info,
                frame_nmr,
                track_ids,
                max_people_car,
                MAX_PEOPLE)

            frame_out_thermal, report_out = create_frame_visual_thermal(
                frame_thermal,
                results,
                frame_nmr,
                max_people_car,
                MAX_PEOPLE)

            if WRITE_VID_THERMAL:
                out_thermal.write(frame_out_thermal)

            with open(live_report_csv_path, 'a') as f:
                if len(report_out) == 0:
                    f.write('{},{},{},{},{}\n'.format(
                        frame_nmr,
                        frame_nmr/fps_thermal,
                        time.time() - frame_start_time, "-", "-"))
                for car in report_out:

                    for entry in car_ids_frames:
                        if entry['car_id'] == car['id']:
                            entry['people'] = car["people"]

                    f.write('{},{},{},{},{}\n'.format(
                        frame_nmr,
                        frame_nmr/fps_thermal,
                        time.time() - frame_start_time, car["id"],
                        car["people"]))
                f.close()

            with open(report_csv, 'a') as f:
                for entry in car_ids_frames:
                    if (frame_nmr - entry['last_seen_frame'] > 10):
                        f.write('{},{},{},{}\n'.format(
                            entry['car_id'],
                            entry['first_seen_frame'],
                            entry['last_seen_frame'],
                            entry['people']))
                        car_ids_frames.remove(entry)
                f.close()

    with open(report_csv, 'a') as f:
        for entry in car_ids_frames:
            f.write('{},{},{},{}\n'.format(
                    entry['car_id'],
                    entry['first_seen_frame'],
                    entry['last_seen_frame'],
                    entry['people']))
            car_ids_frames.remove(entry)
        f.close()

    if WRITE_VID_THERMAL:
        out_thermal.release()
    cap_thermal.release()

    pbar.close()
