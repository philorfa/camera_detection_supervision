import cv2
import numpy as np
from google.cloud import vision
import pytesseract
from pystackreg import StackReg
import math
import re

VEHICLES = [2, 5, 7]
SCALE = 2


def show_img(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)


def detect_car(frame, model, conf):

    car_detections = model(frame, conf=conf, classes=VEHICLES,
                           verbose=False)[0]
    return car_detections


def detect_in_thermal(frame, model, conf):
    prediction = model.predict(frame, confidence=conf).json()
    return prediction['predictions']


def detect_plate(frame, model):
    plate_detections = model(frame, conf=0.4, verbose=False)[0]
    return plate_detections


def crop_detections(frame, coordinates, scale=SCALE):
    x1, y1, x2, y2 = map(int, coordinates[:4])
    region = frame[y1:y2, x1:x2]

    zoomed_in_region = cv2.resize(region,
                                  None,
                                  fx=scale,
                                  fy=scale,
                                  interpolation=cv2.INTER_LANCZOS4)

    return zoomed_in_region


def crop_detections_thermal(frame, coordinates, scale=SCALE):
    x1, y1, x2, y2 = coordinates
    region = frame[y1:y2, x1:x2]

    zoomed_in_region = cv2.resize(region,
                                  None,
                                  fx=scale,
                                  fy=scale,
                                  interpolation=cv2.INTER_LANCZOS4)

    return zoomed_in_region


def qualify_image(original_image, template_url, registration=False):

    # reference plate
    reference_image = cv2.imread(template_url, cv2.IMREAD_ANYCOLOR)

    templateGray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    imageGray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(500)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    if len(matches) < 60:
        return False, []
    else:
        if not registration:
            return True, []

        matches = sorted(matches, key=lambda x: x.distance)
        # keep only the top matches
        keep = int(len(matches) * 0.2)
        matches = matches[:keep]

        # allocate memory for the keypoints (x, y)-coordinates from the
        # top matches -- we'll use these coordinates to compute our
        # homography matrix
        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")
        # loop over the top matches
        for (i, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images
            # map to each other
            ptsA[i] = kpsA[m.queryIdx].pt
            ptsB[i] = kpsB[m.trainIdx].pt

            # compute the homography matrix between the two sets of matched
            # points
            (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
            # use the homography matrix to align the images
            (h, w) = reference_image.shape[:2]
            if H is None:
                return False, []

            aligned = cv2.warpPerspective(original_image, H, (w, h))
            return True, aligned


def google_vision_ocr(image, client):

    text_detection_params = vision.TextDetectionParams(
        enable_text_detection_confidence_score=True)
    image_context = vision.ImageContext(
        text_detection_params=text_detection_params)

    image = vision.Image(content=cv2.imencode('.jpg', image)[1].tostring())
    response = client.text_detection(image=image, image_context=image_context)
    full_text = response.full_text_annotation
    check = full_text.text + "check"
    if check == "check":
        return "Cannot read", 0.0

    # print('\n"{}"'.format(full_text.text))
    # print("confidence: {}".format(full_text.pages[0].confidence))
    text = re.sub(r"[^a-zA-Z0-9 \n\.]", " ", full_text.text)
    return text, full_text.pages[0].confidence


def easyocr_model(image, reader):
    output = reader.readtext(image,
                             allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    text = ''
    confidence = 0
    cnt = len(output)

    if cnt == 0:
        return "Cannot read", 0.0

    for item in output:
        text = ''.join(str(item[1]))
        confidence += item[2]
    final_conf = confidence / cnt
    return text, final_conf


def tesseractocr_model(image):
    predicted_result = pytesseract.image_to_string(
        image,
        lang='eng',
        config=(
            '--oem 3 --psm 7 '
            '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
    print(predicted_result)


def register(ref, template_url):
    mov = cv2.imread(template_url, cv2.IMREAD_ANYCOLOR)
    mov = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    # Translational transformation
    sr = StackReg(StackReg.TRANSLATION)
    out_tra = sr.register_transform(ref, mov)
    show_img(out_tra)

    # Rigid Body transformation
    sr = StackReg(StackReg.RIGID_BODY)
    out_rot = sr.register_transform(ref, mov)
    show_img(out_rot)

    # Scaled Rotation transformation
    sr = StackReg(StackReg.SCALED_ROTATION)
    out_sca = sr.register_transform(ref, mov)
    show_img(out_sca)

    # Affine transformation
    sr = StackReg(StackReg.AFFINE)
    out_aff = sr.register_transform(ref, mov)
    show_img(out_aff)

    # Bilinear transformation
    sr = StackReg(StackReg.BILINEAR)
    out_bil = sr.register_transform(ref, mov)
    show_img(out_bil)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image,
                            rot_mat,
                            image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)
    return result


def compute_skew(src_img):

    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')

    img = cv2.medianBlur(src_img, 3)

    edges = cv2.Canny(img,
                      threshold1=30,
                      threshold2=100,
                      apertureSize=3,
                      L2gradient=True)
    lines = cv2.HoughLinesP(edges,
                            1,
                            math.pi / 180,
                            30,
                            minLineLength=w / 4.0,
                            maxLineGap=h / 4.0)
    angle = 0.0

    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        # print(ang)
        if math.fabs(ang) <= 30:  # excluding extreme rotations
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt) * 180 / math.pi


def deskew(src_img):
    return rotate_image(src_img, compute_skew(src_img))


def preprocess_image(original_image):
    image = deskew(original_image)
    kernel = np.ones((3, 3))

    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 150, 200)
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=2)

    # image = cv2.resize(imageGray, (2100, 1500))

    # histogram equalization
    equ = cv2.equalizeHist(imgThres)

    # manual thresholding
    # th2 = 100 # this threshold might vary!
    # equ[equ>=th2] = 255
    # equ[equ<th2]  = 0
    return equ


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_car(detected_itm, vehicle_track_ids):

    x1, y1, x2, y2 = detected_itm["coords"]

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 >= xcar1 and y1 >= ycar1 and x2 <= xcar2 and y2 <= ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


def get_car_thermal(detected_itm, vehicle_track_ids):

    x1, y1, x2, y2 = detected_itm["coords"]

    foundIt = False

    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        item_detected = [x1, y1, x2, y2]
        car_tracker = [xcar1, ycar1, xcar2, ycar2]
        if bb_intersection_over_union(item_detected, car_tracker) > 0.9:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


def check_resolution(video_path):

    vid = cv2.VideoCapture(video_path)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    return height, width


def convert_coords_to_original_image(bbox, coords, scale=SCALE):
    x1, y1, _, _ = bbox
    new_x1, new_y1, new_x2, new_y2 = coords

    # Scale down the new coordinates by a factor of 2
    old_x1 = x1 + (new_x1 // scale)
    old_y1 = y1 + (new_y1 // scale)
    old_x2 = x1 + (new_x2 // scale)
    old_y2 = y1 + (new_y2 // scale)

    return old_x1, old_y1, old_x2, old_y2


def update_threshold(conf_threshold, car_id, plate):
    found = False
    for sub in conf_threshold:
        if sub['car_id'] == car_id:
            found = True
            if sub['conf'] < plate['conf']:
                sub['conf'] = plate['conf']
                sub['text'] = plate['text']
    if not found:
        conf_threshold.append(
            dict({
                'car_id': car_id,
                'conf': plate['conf'],
                'text': plate['text']
            }))

    return conf_threshold


def update_results(results, plates_coords_texts, frame_nmr, track_ids,
                   conf_threshold, max_condidence):
    for pl in plates_coords_texts:
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(pl, track_ids)
        if car_id != -1:

            x1, y1, x2, y2 = pl["coords"]
            results[frame_nmr][car_id] = {
                'car': {
                    'bbox': [xcar1, ycar1, xcar2, ycar2]
                },
                'license_plate': {
                    'bbox': [x1, y1, x2, y2],
                    'text': pl['text'],
                    'bbox_score': pl['bbox_score'],
                    'text_score': pl['conf']
                }
            }
            if max_condidence:
                conf_threshold = update_threshold(conf_threshold, car_id, pl)
    return results, conf_threshold


def update_people(max_people_car, car_id, car):
    found = False
    for sub in max_people_car:
        if sub['car_id'] == car_id:
            found = True
            if sub['people_counting'] < car['people']:
                sub['people_counting'] = car['people']
    if not found:
        max_people_car.append(
            dict({
                'car_id': car_id,
                'people_counting': car['people']
            }))

    return max_people_car


def update_results_thermal(results, car_info, frame_nmr, track_ids,
                           max_people_car, max_people):

    for car in car_info:
        xcar1, ycar1, xcar2, ycar2, car_id = get_car_thermal(car, track_ids)

        if car_id != -1:

            x1, y1, x2, y2 = car["coords"]
            results[frame_nmr][car_id] = {
                'car': {
                    'bbox': [xcar1, ycar1, xcar2, ycar2]
                },
                'people': {
                    'number': car['people'],
                    'coordinates': car['people_coords'],
                    'confidence': car['conf']
                }
            }
            if max_people:
                max_people_car = update_people(max_people_car, car_id, car)
    return results, max_people_car


def search_threshold(conf_threshold, id):
    target_id = id

    car_list = [d['car_id'] for d in conf_threshold]
    try:
        index = car_list.index(target_id)
    except ValueError:
        index = None

    if index is not None:
        res = conf_threshold[index]
    else:
        res = None

    return res


def create_frame_visual(frame, results, frame_nmr, conf_threshold,
                        max_confidence):
    show = False
    report = []
    for plate in results[frame_nmr]:

        show = True
        car_x1, car_y1, car_x2, car_y2 = results[frame_nmr][plate]["car"][
            "bbox"]
        pl_x1, pl_y1, pl_x2, pl_y2 = results[frame_nmr][plate][
            "license_plate"]["bbox"]

        if max_confidence:
            info = search_threshold(conf_threshold, plate)
        else:
            info = None

        if info is not None:
            text = info["text"]
            confidence = info["conf"]
        else:
            text = results[frame_nmr][plate]["license_plate"]["text"]
            confidence = results[frame_nmr][plate]["license_plate"][
                "text_score"]

        cv2.rectangle(frame, (int(car_x1), int(car_y1)),
                      (int(car_x2), int(car_y2)), (0, 0, 255), 2)

        if confidence > 0.8:
            license_color = (0, 255, 0)
        else:
            license_color = (0, 0, 255)

        cv2.rectangle(frame, (int(pl_x1), int(pl_y1)),
                      (int(pl_x2), int(pl_y2)), license_color, 2)

        # Add text above the rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 2

        cv2.putText(frame, text, (int(pl_x1), int(pl_y2 + 40)), font,
                    font_scale, license_color, thickness)
        report_frame = {"id": plate, "number": text, "conf": confidence}
        report.append(dict(report_frame))

    return frame, show, report


def get_max_people(car_id, max_people_car):
    target_id = car_id

    car_list = [d['car_id'] for d in max_people_car]
    try:
        index = car_list.index(target_id)
    except ValueError:
        index = None

    if index is not None:
        res = max_people_car[index]["people_counting"]
    else:
        res = None

    return res


def create_frame_visual_thermal(frame_thermal, results, frame_nmr,
                                max_people_car, max_people):

    report = []
    people_color = (0, 255, 0)
    car_color = (0, 0, 255)
    for car in results[frame_nmr]:

        car_x1, car_y1, car_x2, car_y2 = results[frame_nmr][car]["car"]["bbox"]
        cv2.rectangle(frame_thermal, (int(car_x1), int(car_y1)),
                      (int(car_x2), int(car_y2)), car_color, 2)

        if not max_people:

            people_in_car = results[frame_nmr][car]["people"]["number"]
            if people_in_car > 0:
                people_coords = (
                    results[frame_nmr][car]["people"]["coordinates"])

                for people in people_coords:
                    cv2.rectangle(frame_thermal,
                                  (int(people[0]), int(people[1])),
                                  (int(people[2]), int(people[3])),
                                  people_color, 2)

        else:
            people_in_car = get_max_people(car, max_people_car)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 2
        cv2.putText(frame_thermal, str(people_in_car),
                    (int(car_x1), int(car_y1 + 40)), font, font_scale,
                    people_color, thickness)

        report_frame = {"id": car, "people": people_in_car}
        report.append(dict(report_frame))
    return frame_thermal, report


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id',
                                                'car_bbox',
                                                'license_plate_bbox',
                                                'license_plate_bbox_score',
                                                'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                # print(results[frame_nmr][car_id])
                if ('car' in results[frame_nmr][car_id].keys() and
                        'license_plate' in results[frame_nmr][car_id].keys()
                        and 'text'
                        in results[frame_nmr][car_id]['license_plate'].keys()):
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr, car_id, '[{} {} {} {}]'.format(
                            results[frame_nmr][car_id]['car']['bbox'][0],
                            results[frame_nmr][car_id]['car']['bbox'][1],
                            results[frame_nmr][car_id]['car']['bbox'][2],
                            results[frame_nmr][car_id]['car']['bbox'][3]),
                        '[{} {} {} {}]'.format(
                            results[frame_nmr][car_id]['license_plate']['bbox']
                            [0], results[frame_nmr][car_id]['license_plate']
                            ['bbox'][1], results[frame_nmr][car_id]
                            ['license_plate']['bbox'][2], results[frame_nmr]
                            [car_id]['license_plate']['bbox'][3]),
                        results[frame_nmr][car_id]['license_plate']
                        ['bbox_score'],
                        results[frame_nmr][car_id]['license_plate']['text'],
                        results[frame_nmr][car_id]['license_plate']
                        ['text_score']))
        f.close()


def draw_yolo_bounding_box(image, coordinates):
    x1, y1, x2, y2 = map(int, coordinates[:4])

    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Convert BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
