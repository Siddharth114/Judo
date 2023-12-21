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
            results = model.predict(frame, classes=[0], show_conf=False, show_labels=False)

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
    print(fps)
    # video_name = "static/imgs/walking_vid_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # height, width = frames[0].shape[0], frames[0].shape[1]

    frame_size=(width, height)

    # print(frames[0].shape)

    writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for frame in frames:
        writer.write(frame)

    writer.release()

def translate_coords(mouse_x, mouse_y, vid_width, vid_height, original_width, original_height):
    new_x = mouse_x * (original_width / vid_width)
    new_y = mouse_y * (original_height / vid_height)

    return new_x, new_y


def check_inside_box(bounding_boxes, x, y):
    for box in bounding_boxes:
        print(box)
        x_tl, y_tl, x_br, y_br = box
        if (x_tl <= x <= x_br) and (y_tl <= y <= y_br):
            return box
    return False

def cropped_img(frame, boxes, x, y):
    inside_box = check_inside_box(boxes, x, y)
    if not inside_box:
        return frame
    
    xmin, ymin, xmax, ymax = inside_box
    xmin = max(0,xmin-50)
    ymin = max(0, ymin-50)
    xmax = min(frame.shape[1],xmax+50)
    ymax = min(frame.shape[0],ymax+50)
    
    cropped_image = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

    return cropped_image


def matching_frame(frames, frame):
    for list_frame in frames:
        if np.all(frame == list_frame):
            return True
    return False



if __name__=='__main__':
    # frames, bounding_boxes, width, height = get_frames('starter_images/walking_vid.mp4')
    pass