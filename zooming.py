from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

def draw_boxes(img, coords):
    for i in coords:
        (x1,y1,x2,y2,) = i
        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))

        img = cv2.rectangle(
            img, start_point, end_point, color=(0,0,255), thickness=3
        )
    # cv2.imshow('Person Detection', img)
    return img


def get_frames(video_path):
    model = YOLO('yolov8n.pt')
    frames=[]
    bounding_boxes = []
    # video_path = "starter_images/walking_vid.mp4"

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model.predict(frame, classes=[0], show_conf=False, show_labels=False, verbose=False)

            bounding_boxes.append(results[0].boxes.xyxy.tolist())

            # Visualize the results on the frame
            annotated_frame = draw_boxes(frame, bounding_boxes[-1])

            frames.append(annotated_frame)

            # Display the annotated frame
            # cv2.imshow("Person Detection", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    height, width = frames[0].shape[0], frames[0].shape[1]

    return frames, bounding_boxes, width, height, original_fps

def write_video(frames, video_path, width, height, fps):
    # print(fps)
    # video_name = "static/imgs/walking_vid_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # height, width = frames[0].shape[0], frames[0].shape[1]

    frame_size=(width, height)

    # print(frames[0].shape)

    writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for frame in frames:
        frame_to_write = fit_to_resolution(frame, width, height)
        writer.write(frame_to_write)

    writer.release()

def translate_coords(mouse_x, mouse_y, vid_width, vid_height, original_width, original_height):
    new_x = mouse_x * (original_width / vid_width)
    new_y = mouse_y * (original_height / vid_height)

    return new_x, new_y

def check_inside_box(bounding_boxes, x, y): #returns the box that the coordinate is inside
    for box in bounding_boxes:
        x_tl, y_tl, x_br, y_br = box
        if (x_tl <= x <= x_br) and (y_tl <= y <= y_br):
            return box
    return False

def cropped_img(frame, boxes, x, y): #current frame, list of boxes for each frame, coordinates. returns cropped image
    inside_box = check_inside_box(boxes, x, y)
    if not inside_box:
        return frame, boxes[0][0]
    
    xmin, ymin, xmax, ymax = inside_box
    xmin = max(0,xmin-50)
    ymin = max(0, ymin-50)
    xmax = min(frame.shape[1],xmax+50)
    ymax = min(frame.shape[0],ymax+50)
    
    cropped_image = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

    return cropped_image, inside_box

def matching_frame(frames, frame):
    for list_frame in frames:
        if np.all(frame == list_frame):
            return True
    return False

def generate_cropped_frames(frames, boxes, box_to_track):
    cropped_frames = [crop_box_from_frame(frames[0], box_to_track)]
    prev_box = box_to_track
    for ind, (curr_frame, next_frame) in enumerate(zip(frames, frames[1:])):
        next_box = get_next_frame_box(prev_box, boxes[ind])
        prev_box = next_box
        cropped_frames.append(crop_box_from_frame(next_frame, next_box))

    return cropped_frames


def crop_box_from_frame(frame, box):
    xmin, ymin, xmax, ymax = box
    xmin = max(0,xmin-50)
    ymin = max(0, ymin-50)
    xmax = min(frame.shape[1],xmax+50)
    ymax = min(frame.shape[0],ymax+50)

    cropped_image = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
    return cropped_image



def get_next_frame_box(box_to_track, boxes):
    threshold_distance = 10
    x1,y1,x2,y2 = box_to_track
    center1 = ((x2+x1)/2, (y2+y1)/2)
    for box in boxes:
        x3,y3,x4,y4 = box
        center2 = ((x4+x3)/2, (y4+y3)/2)

        distance = ( (center2[0]-center1[0])**2 + (center2[1]-center1[1])**2 )**.5
        if distance>threshold_distance:
            continue
        return box
    return box_to_track #returns old frame if no new one is detected


def fit_to_resolution(frame, width, height):

    frame_height, frame_width = frame.shape[:2]

    if frame_height < height or frame_width < width:  # Frame is smaller, pad with black bars
        result = np.zeros((height, width, 3), dtype=frame.dtype)  # Create black canvas
        start_y = (height - frame_height) // 2
        start_x = (width - frame_width) // 2
        result[start_y:start_y+frame_height, start_x:start_x+frame_width] = frame
    elif frame_height > height or frame_width > width:  # Frame is bigger, downsize
        result = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    else:  # Frame is already the correct size, return as is
        result = frame

    return result

if __name__=='__main__':
    # frames, bounding_boxes, width, height = get_frames('starter_images/walking_vid.mp4')
    pass