from ultralytics import YOLO
import cv2
import numpy as np
import glob
import os
from itertools import combinations
import copy

# drawing bounding boxes given the coordinates from the yolo model
def draw_boxes(img, coords):
    # iterating over each of the boxes and getting the coordinates of the bounding boxes
    for i in coords:
        (x1,y1,x2,y2,) = i

        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))
        # drawing a rectangle in the coordinates where the bounding box is supposed to be. 
        # the original drawing using yolov8 has the class label which is not required
        img = cv2.rectangle(
            img, start_point, end_point, color=(0,0,255), thickness=3
        )
    return img

def get_boxes(boxes):
    
    merged = []
    t_boxes = copy.deepcopy(boxes)

    boxes_to_check = merged + t_boxes

    while any(mergeable(j0,k0) for j0,k0 in combinations(boxes_to_check, r=2)):
        merged = []
        t_boxes = copy.deepcopy(boxes_to_check)
        for j,k in combinations(boxes_to_check, r=2):
            if mergeable(j,k):
                merged.append(list(merge(j,k)))
                try:
                    t_boxes.remove(j)
                except ValueError:
                    pass
                try:
                    t_boxes.remove(k)
                except ValueError:
                    pass
                break
        boxes_to_check = t_boxes + merged
    
    return boxes_to_check

def mergeable(box1, box2, x_val=100, y_val=100):
    (
        x1,
        y1,
        x2,
        y2,
    ) = box1
    (
        x3,
        y3,
        x4,
        y4,
    ) = box2

    intersection_check = not (
        (x3>x_val+x2 or x4+x_val<x1) or (y3>y2+y_val or y4+y_val<y1)
    )


    w1 = x2-x1
    w2 = x4-x3
    h1 = y2-y1
    h2 = y4-y3

    dimension_check = not ( (w1<.75*w2 and h1<.75*h2) or (w2<.75*w1 and h2<.75*h1))

    area1 = w1*h1
    area2 = w2*h2

    # area_check = not ( (area1<.5*area2) or (area2<.5*area1) )
    
    return intersection_check and dimension_check

def merge(box1, box2):
    (
        x1,
        y1,
        x2,
        y2,
    ) = box1
    (
        x3,
        y3,
        x4,
        y4,
    ) = box2

    x_min = min(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    x_max = max(x1, x2, x3, x4)
    y_max = max(y1, y2, y3, y4)

    return x_min, y_min, x_max, y_max

def get_frames(video_path):
    model = YOLO("yolov8n.pt")
    frames=[]
    final_boxes = []

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.predict(frame, classes=[0], augment=False, show_conf=False, show_labels=False, conf=0.8, verbose=False)
            
            final_boxes.append(get_boxes(results[0].boxes.xyxy.tolist()))

            annotated_frame = draw_boxes(frame, final_boxes[-1])

            frames.append(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    height, width = frames[0].shape[0], frames[0].shape[1]
    return frames, final_boxes, width, height, original_fps

def write_video(frames, video_path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    frame_size = (width, height)

    writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for frame in frames:
        frame_to_write = fit_to_resolution(frame, width, height)
        writer.write(frame_to_write)

    writer.release()

def fit_to_resolution(frame, width, height):
    frame_height, frame_width = frame.shape[:2]
    frame_aspect_ratio = frame_width / frame_height  # Calculate original aspect ratio
    target_aspect_ratio = width / height             # Calculate target aspect ratio

    # Determine scaling factor based on aspect ratios:
    if frame_aspect_ratio > target_aspect_ratio:
        scaling_factor = width / frame_width
    else:
        scaling_factor = height / frame_height

    # Scale the frame while preserving aspect ratio:
    scaled_width = int(frame_width * scaling_factor)
    scaled_height = int(frame_height * scaling_factor)
    scaled_frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    # Create a black canvas with the target dimensions:
    result = np.zeros((height, width, 3), dtype=scaled_frame.dtype)

    # Center the scaled frame within the canvas:
    start_y = (height - scaled_height) // 2
    start_x = (width - scaled_width) // 2
    result[start_y: start_y + scaled_height, start_x: start_x + scaled_width] = scaled_frame

    return result

def translate_coords(
    mouse_x, mouse_y, vid_width, vid_height, original_width, original_height
):
    # calculating new coordinates of the mouse click to adjust for the change in size of the video
    new_x = mouse_x * (original_width / vid_width)
    new_y = mouse_y * (original_height / vid_height)

    return new_x, new_y

def check_inside_box(
    bounding_boxes, x, y
):  
    # iterating through all the boxes of the frame and returning which box the mouse click is inside
    for box in bounding_boxes:
        x_tl, y_tl, x_br, y_br = box
        if (x_tl <= x <= x_br) and (y_tl <= y <= y_br):
            return box
    return False

def cropped_img(
    frame, boxes, x, y
):
    # first checking if the mouse click is inside a bounding box
    inside_box = check_inside_box(boxes, x, y)
    # if the mouse click is not inside the box, return the frame with the first box in the list of boxes
    if not inside_box:
        return frame, boxes[0][0], False

    # getting coordinates of the bounding box
    xmin, ymin, xmax, ymax = inside_box
    xmin = max(0, xmin - 50)
    ymin = max(0, ymin - 50)
    xmax = min(frame.shape[1], xmax + 50)
    ymax = min(frame.shape[0], ymax + 50)

    # cropping the image with some padding around the person
    cropped_image = frame[int(ymin) : int(ymax), int(xmin) : int(xmax)]

    return cropped_image, inside_box, True

def generate_cropped_frames(frames, boxes, box_to_track):
    # initializing a list of frames which are zooming in on the box to track
    cropped_frames = [crop_box_from_frame(frames[0], box_to_track)]
    prev_box = box_to_track
    for ind, (curr_frame, next_frame) in enumerate(zip(frames, frames[1:])):
        next_box = get_next_frame_box(prev_box, boxes[ind])
        prev_box = next_box
        cropped_frames.append(crop_box_from_frame(next_frame, next_box))

    return cropped_frames

def crop_box_from_frame(frame, box):
    xmin, ymin, xmax, ymax = box
    xmin = max(0, xmin - 50)
    ymin = max(0, ymin - 50)
    xmax = min(frame.shape[1], xmax + 50)
    ymax = min(frame.shape[0], ymax + 50)

    cropped_image = frame[int(ymin) : int(ymax), int(xmin) : int(xmax)]
    return cropped_image

def get_next_frame_box(box_to_track, boxes):
    threshold_distance = 50
    x1, y1, x2, y2 = box_to_track
    center1 = ((x2 + x1) / 2, (y2 + y1) / 2)
    for box in boxes:
        x3, y3, x4, y4 = box
        center2 = ((x4 + x3) / 2, (y4 + y3) / 2)

        distance = (
            (center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2
        ) ** 0.5
        if distance > threshold_distance:
            continue
        return box
    return box_to_track

def wildcard_delete(pattern):
    matching_files = glob.glob(pattern)
    for file in matching_files:
        os.remove(file)
