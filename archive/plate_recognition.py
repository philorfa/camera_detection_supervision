from ultralytics import YOLO
import cv2

from util import detect_car, detect_plate, crop_detections, google_vision_ocr
from util import show_img, convert_coords_to_original_image
from util import write_csv, update_results, create_frame_visual

import os
import easyocr

import numpy as np
from google.cloud import vision

import sys
from pathlib import Path

from tqdm import tqdm

import time
import yaml

sys.path.append(os.path.abspath(Path.cwd()))

try:
    from sort.sort import Sort
except (Exception, ):
    raise Exception('You need sort module!')


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

google_credentials_path = config['google_application_credentials']

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = (google_credentials_path)

VID_FOLDER = 4

WRITE_VID_RGB = True
SHOW_FRAMES = False
MAX_CONFIDENCE = False

if __name__ == '__main__':
    print("> Starting Initialization ...")
    init_start_time = time.time()

    # report folder
    report_folder = os.path.join('./reports', str(VID_FOLDER))
    os.makedirs(report_folder, exist_ok=True)

    # template for finding matches
    template_url = "./data/plate_reference_1.png"

    # load object detection models
    car_model = YOLO('./models/yolov8n.pt')
    plate_model = YOLO('./models/license_plate_detector.pt')

    # load ocr models
    google_client = vision.ImageAnnotatorClient()
    reader = easyocr.Reader(['en'], gpu=True)

    # load tracker
    car_tracker = Sort()

    # load video
    rgb_video = 'rgb.MOV'
    rgb_video_path = os.path.join('./rgb_data', str(VID_FOLDER), rgb_video)

    cap_rgb = cv2.VideoCapture(rgb_video_path)

    # output csv
    license_csv = os.path.join(report_folder, 'license_plate.csv')

    report_csv = os.path.join(report_folder, 'output_report.csv')

    # live report csv
    if MAX_CONFIDENCE:
        live_report_csv_path = os.path.join(report_folder,
                                            "report_maxconf.csv")
    else:
        live_report_csv_path = os.path.join(report_folder, "report_live.csv")

    with open(live_report_csv_path, 'w') as f:
        f.write('{},{},{},{},{},{}\n'.format('frame_nmr', 'sec_in_vid(sec)',
                                             'frame_time_processed(sec)',
                                             'car_id', 'license_number',
                                             'license_number_conf'))
        f.close()

    with open(report_csv, 'w') as f:
        f.write('{},{},{},{},{}\n'.format('car_id',
                                          'first_frame',
                                          'last_frame',
                                          'plate_number',
                                          'plate_number_conf'))
        f.close()

    # output live rgb video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps_rgb = cap_rgb.get(cv2.CAP_PROP_FPS)
    width_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if WRITE_VID_RGB:
        if MAX_CONFIDENCE:
            output_live_vid = os.path.join(report_folder, 'rgb_live_conf.mp4')
        else:
            output_live_vid = os.path.join(report_folder, 'rgb_live.mp4')

        out_rgb = cv2.VideoWriter(output_live_vid, fourcc, fps_rgb,
                                  (width_rgb, height_rgb))

        # check video resolution
        print("--- RGB Video info ---")
        print("Resolution width x height: ", int(width_rgb), " x ",
              int(height_rgb))
        print("FPS ", fps_rgb)

        length_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total Frames: ", length_rgb)

    results = {}
    conf_threshold = []

    car_ids_frames = []

    # read frames
    frame_nmr = -1
    ret_rgb = True

    frame_count = length_rgb

    init_end_time = time.time() - init_start_time
    print("> Initialization time(sec) :", init_end_time)
    pbar = tqdm(total=frame_count)

    while ret_rgb and frame_nmr < frame_count:

        frame_start_time = time.time()

        frame_nmr += 1
        ret_rgb, frame_rgb = cap_rgb.read()

        pbar.update(1)

        if ret_rgb and frame_nmr < frame_count:

            results[frame_nmr] = {}

            car_detections = detect_car(frame_rgb, car_model, conf=0.4)
            tracker_detections = []
            plates_coords_texts = []
            car_detected = False

            for car in car_detections.boxes.data.tolist():

                car_detected = True
                tracker_detections.append(car[:5])
                zoomed_in_car_rgb = crop_detections(frame_rgb, car)

                plate_detections = detect_plate(zoomed_in_car_rgb, plate_model)

                if len(plate_detections) > 2:
                    raise Exception("Found 2 plates for 1 car")
                elif len(plate_detections) == 1:
                    plate = plate_detections.boxes.data.tolist()[0]
                    old_pl_coords = convert_coords_to_original_image(
                        car[:4], plate[:4])

                    zoomed_in_plate = crop_detections(zoomed_in_car_rgb, plate)

                    image = zoomed_in_plate

                    text, conf = google_vision_ocr(image, google_client)

                    coords_text = {
                        'coords': old_pl_coords,
                        'text': text,
                        'bbox_score': plate[4],
                        'conf': conf
                    }
                    plates_coords_texts.append(dict(coords_text))

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
                                     'plate': "Cannot read",
                                     'conf': 0.0}
                        car_ids_frames.append(new_entry)

            results, conf_threshold = update_results(results,
                                                     plates_coords_texts,
                                                     frame_nmr, track_ids,
                                                     conf_threshold,
                                                     MAX_CONFIDENCE)

            frame_out_rgb, found, report_out = create_frame_visual(
                frame_rgb, results, frame_nmr, conf_threshold, MAX_CONFIDENCE)

            if WRITE_VID_RGB:
                out_rgb.write(frame_out_rgb)

            if found and SHOW_FRAMES:
                show_img(frame_out_rgb)

            with open(live_report_csv_path, 'a') as f:
                if len(report_out) == 0:
                    f.write('{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        frame_nmr/fps_rgb,
                        time.time() - frame_start_time, "-", "-", "-"))
                for car in report_out:

                    for entry in car_ids_frames:
                        if entry['car_id'] == car['id']:
                            entry['plate'] = car["number"]
                            entry['conf'] = car['conf']

                    f.write('{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        frame_nmr/fps_rgb,
                        time.time() - frame_start_time, car["id"],
                        car["number"], car["conf"]))
                f.close()

            with open(report_csv, 'a') as f:
                for entry in car_ids_frames:
                    if (frame_nmr - entry['last_seen_frame'] > 5 and
                            entry["conf"] > 0.9):
                        f.write('{},{},{},{},{}\n'.format(
                            entry['car_id'],
                            entry['first_seen_frame'],
                            entry['last_seen_frame'],
                            entry['plate'],
                            entry['conf']))
                        car_ids_frames.remove(entry)
                f.close()

    with open(report_csv, 'a') as f:
        for entry in car_ids_frames:
            if (entry["conf"] > 0.9):
                f.write('{},{},{},{},{}\n'.format(
                    entry['car_id'],
                    entry['first_seen_frame'],
                    entry['last_seen_frame'],
                    entry['plate'],
                    entry['conf']))
                car_ids_frames.remove(entry)
        f.close()

    write_csv(results, license_csv)
    if WRITE_VID_RGB:
        out_rgb.release()
    cap_rgb.release()

    pbar.close()
