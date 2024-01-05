from ultralytics import YOLO
import cv2
import numpy as np
import glob
import os
from collections import defaultdict

def draw_boxes(frame, boxes):
    for i in boxes:
        xyxy, id = i
        x1, y1, x2, y2 = xyxy
        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))
        frame = cv2.rectangle(frame, start_point, end_point, color=(0, 0, 255), thickness=3)
        frame = cv2.putText(frame, str(int(id)), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) #writing id number above box
    return frame

def get_frames(video_path):
    frames=[]
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    final_bounding_boxes = []
    track_history = defaultdict(lambda: [])
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True, verbose=False)
            bounding_boxes = []
            xyxy = results[0].boxes.xyxy.tolist()
            id = results[0].boxes.id.tolist()
            for i,j in zip(xyxy, id):
                bounding_boxes.append((i, j))
            final_bounding_boxes.append(bounding_boxes)
            annotated_frame = draw_boxes(frame, bounding_boxes)
            
            # tracking lines for each person
            for box, track_id in zip(results[0].boxes.xywh.cpu(), results[0].boxes.id.int().cpu().tolist()):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 215, 255), thickness=3)

            frames.append(annotated_frame)


            # Display the annotated frame
            # cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    height, width = frames[0].shape[0], frames[0].shape[1]
    return frames, final_bounding_boxes, width, height, original_fps

def write_video(frames, video_path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    frame_size = (width, height)

    writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for frame in frames:
        frame_to_write = fit_to_resolution(frame, width, height)
        writer.write(frame_to_write)

    writer.release()

def translate_coords(
    mouse_x, mouse_y, vid_width, vid_height, original_width, original_height
):
    # calculating new coordinates of the mouse click to adjust for the change in size of the video
    new_x = mouse_x * (original_width / vid_width)
    new_y = mouse_y * (original_height / vid_height)

    return new_x, new_y

def check_inside_box(bounding_boxes, x, y):
    for box, id in bounding_boxes:
        x_tl, y_tl, x_br, y_br = box
        if (x_tl <= x <= x_br) and (y_tl <= y <= y_br):
            return id
    return False

def generate_cropped_frames(frames, boxes, id_to_track):
    cropped_frames = []
    for frame, boxes_per_frame in zip(frames, boxes):
        for box, id in boxes_per_frame:
            if id==id_to_track:
                prev_coords = box
                cropped_frames.append(crop_box_from_frame(frame, box))
                break
        # new adds
        else:
            cropped_frames.append(crop_box_from_frame(frame, prev_coords))

    return cropped_frames

def crop_box_from_frame(frame, box):
    xmin, ymin, xmax, ymax = box
    xmin = max(0, xmin - 50)
    ymin = max(0, ymin - 50)
    xmax = min(frame.shape[1], xmax + 50)
    ymax = min(frame.shape[0], ymax + 50)

    cropped_image = frame[int(ymin) : int(ymax), int(xmin) : int(xmax)]
    return cropped_image

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

def wildcard_delete(pattern):
    matching_files = glob.glob(pattern)
    for file in matching_files:
        os.remove(file)