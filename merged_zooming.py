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
    '''
    Determines whether two rectangular bounding boxes can be merged based on their spatial proximity and dimensions.
    Args:
    box1 (tuple): A tuple representing the coordinates of the first box in the format (x1, y1, x2, y2).
    box2 (tuple): A tuple representing the coordinates of the second box in the format (x3, y3, x4, y4).
    x_val (int, optional): An optional value used for adjusting the intersection check along the x-axis. Defaults to 100.
    y_val (int, optional): An optional value used for adjusting the intersection check along the y-axis. Defaults to 100.

    Returns:
    bool: True if the boxes can be merged, False otherwise.
    '''
    # Unpack box coordinates
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
    # Intersection check: Ensure boxes have a significant overlap
    intersection_check = not (
        (x3>x_val+x2 or x4+x_val<x1) or (y3>y2+y_val or y4+y_val<y1)
    )

    # Dimension check: Ensure boxes have similar dimensions
    w1 = x2-x1
    w2 = x4-x3
    h1 = y2-y1
    h2 = y4-y3

    dimension_check = not ( (w1<.75*w2 and h1<.75*h2) or (w2<.75*w1 and h2<.75*h1))

    # Area check (commented out): Could be used for further refinement
    # area1 = w1*h1
    # area2 = w2*h2

    # area_check = not ( (area1<.5*area2) or (area2<.5*area1) )
    
    # Require both intersection and dimension checks to pass
    return intersection_check and dimension_check

def merge(box1, box2):
    '''
    Merges two rectangular bounding boxes into a single, larger bounding box that encompasses both.
    Args:
    box1 (tuple): A tuple representing the coordinates of the first box in the format (x1, y1, x2, y2).
    box2 (tuple): A tuple representing the coordinates of the second box in the format (x3, y3, x4, y4).

    Returns:
    tuple: A tuple representing the coordinates of the merged box in the format (x_min, y_min, x_max, y_max).
    '''
    # Unpack box coordinates
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

    # Find the minimum and maximum coordinates across both boxes
    x_min = min(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    x_max = max(x1, x2, x3, x4)
    y_max = max(y1, y2, y3, y4)

    # Return the coordinates of the merged box
    return x_min, y_min, x_max, y_max

def get_frames(video_path):
    '''
    Extracts frames from a video, applies object detection with YOLOv8, and annotates the frames with bounding boxes.
    Args:
    video_path (str): The path to the video file to be processed.

    Returns:
    tuple: A tuple containing:
        - frames (list): A list of annotated video frames as NumPy arrays.
        - final_boxes (list): A list of lists, where each inner list represents the detected object boxes for a corresponding frame.
          Each box is a list of coordinates in the format [x1, y1, x2, y2].
        - width (int): The width of the video frames in pixels.
        - height (int): The height of the video frames in pixels.
        - original_fps (float): The original frames per second of the video.
    '''
    model = YOLO("yolov8n.pt")
    frames=[]
    final_boxes = []
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    # Process each frame
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Perform object detection
            results = model.predict(frame, classes=[0], augment=False, show_conf=False, show_labels=False, conf=0.8, verbose=False)
            # Process bounding boxes
            final_boxes.append(get_boxes(results[0].boxes.xyxy.tolist()))
            # Annotate frame with boxes
            annotated_frame = draw_boxes(frame, final_boxes[-1])
            # Append annotated frame
            frames.append(annotated_frame)
            # Check for early termination (press "q")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    # Get frame dimensions
    height, width = frames[0].shape[0], frames[0].shape[1]
    return frames, final_boxes, width, height, original_fps

def write_video(frames, video_path, width, height, fps):
    '''
    Writes a sequence of frames to a video file using OpenCV.
    Args:
        frames (list): A list of frames to be written, as NumPy arrays.
        video_path (str): The path to the output video file.
        width (int): The width of the video frames in pixels.
        height (int): The height of the video frames in pixels.
        fps (float): The desired frames per second of the output video.
    '''
    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    frame_size = (width, height)

    writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for frame in frames:
        frame_to_write = fit_to_resolution(frame, width, height)
        writer.write(frame_to_write)

    writer.release()

def fit_to_resolution(frame, width, height):
    '''
    Resizes a frame to fit within a specified resolution while preserving its aspect ratio.
    Args:
        frame (np.ndarray): The frame to be resized, as a NumPy array.
        width (int): The desired width of the resized frame.
        height (int): The desired height of the resized frame.

    Returns:
        np.ndarray: The resized frame, centered within a black canvas of the target dimensions.
    '''
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
    '''
    Finds the corresponding box in the next frame that is most likely to represent the same object as the given box, using a threshold distance of 50 pixels.
    Args:
        box_to_track (tuple): A tuple representing the coordinates of the box to track in the format (x1, y1, x2, y2).
        boxes (list): A list of boxes in the next frame, each represented as a tuple (x3, y3, x4, y4).
    Returns:
        tuple: The coordinates of the box in the next frame that is closest to the box to track,
            or the original box_to_track if no close match is found.
    '''
    threshold_distance = 50 # Maximum distance between box centers to consider a match
    # Calculate the center of the box to track
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
    '''
    Deletes files matching a wildcard pattern.
    Args:
        pattern (str): The wildcard pattern to match files against, e.g., "*.txt" or "temp_*.py".

    Passes if not matching files are found
    '''
    matching_files = glob.glob(pattern)
    for file in matching_files:
        os.remove(file)
