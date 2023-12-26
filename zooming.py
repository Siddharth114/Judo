from ultralytics import YOLO
import cv2
import numpy as np

# drawing bounding boxes given the coordinates from the yolo model
def draw_boxes(img, coords):
    # iterating over each of the boxes and getting the coordinates of the bounding boxes
    for i in coords:
        (
            x1,
            y1,
            x2,
            y2,
        ) = i
        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))
        # drawing a rectangle in the coordinates where the bounding box is supposed to be. the original drawing using yolov8 has the class label which is not required
        img = cv2.rectangle(img, start_point, end_point, color=(0, 0, 255), thickness=3)
    # cv2.imshow('Person Detection', img)
    return img


# get frames of video with people detected
def get_frames(video_path):
    # initializing the model
    model = YOLO("yolov8n.pt")
    frames = []
    bounding_boxes = []

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model.predict(
                frame, classes=[0], show_conf=False, show_labels=False, verbose=False
            )

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
    # return data related to the video back to the flask app file
    return frames, bounding_boxes, width, height, original_fps


# multi-purpose video writing funtion with cv2
def write_video(frames, video_path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    frame_size = (width, height)

    writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for frame in frames:
        frame_to_write = fit_to_resolution(frame, width, height)
        writer.write(frame_to_write)

    writer.release()


# translating coordinates to adjust for the size change of the video
def translate_coords(
    mouse_x, mouse_y, vid_width, vid_height, original_width, original_height
):
    # calculating new coordinates of the mouse click to adjust for the change in size of the video
    new_x = mouse_x * (original_width / vid_width)
    new_y = mouse_y * (original_height / vid_height)

    return new_x, new_y


#check if the clicked point is inside a bounding box
def check_inside_box(
    bounding_boxes, x, y
):  
    # iterating through all the boxes of the frame and returning which box the mouse click is inside
    for box in bounding_boxes:
        x_tl, y_tl, x_br, y_br = box
        if (x_tl <= x <= x_br) and (y_tl <= y <= y_br):
            return box
    return False


#returns cropped image given the clicked point and list of boxes for that frame
def cropped_img(
    frame, boxes, x, y
):
    # first checking if the mouse click is inside a bounding box
    inside_box = check_inside_box(boxes, x, y)
    # if the mouse click is not inside the box, return the frame with the first box in the list of boxes
    if not inside_box:
        return frame, boxes[0][0]

    # getting coordinates of the bounding box
    xmin, ymin, xmax, ymax = inside_box
    xmin = max(0, xmin - 50)
    ymin = max(0, ymin - 50)
    xmax = min(frame.shape[1], xmax + 50)
    ymax = min(frame.shape[0], ymax + 50)

    # cropping the image with some padding around the person
    cropped_image = frame[int(ymin) : int(ymax), int(xmin) : int(xmax)]

    return cropped_image, inside_box


# generating cropped frames where the frame is zoomed in on the selected person
def generate_cropped_frames(frames, boxes, box_to_track):
    # initializing a list of frames which are zooming in on the box to track
    cropped_frames = [crop_box_from_frame(frames[0], box_to_track)]
    prev_box = box_to_track
    for ind, (curr_frame, next_frame) in enumerate(zip(frames, frames[1:])):
        next_box = get_next_frame_box(prev_box, boxes[ind])
        prev_box = next_box
        cropped_frames.append(crop_box_from_frame(next_frame, next_box))

    return cropped_frames


# crop the selected bounding box from the frame with some padding around it
def crop_box_from_frame(frame, box):
    xmin, ymin, xmax, ymax = box
    xmin = max(0, xmin - 50)
    ymin = max(0, ymin - 50)
    xmax = min(frame.shape[1], xmax + 50)
    ymax = min(frame.shape[0], ymax + 50)

    cropped_image = frame[int(ymin) : int(ymax), int(xmin) : int(xmax)]
    return cropped_image


# get the box which is closest to the box from the previous frame
def get_next_frame_box(box_to_track, boxes):
    threshold_distance = 20
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
    return box_to_track  # returns old frame if no new one is detected


# given the cropped image, return an image padded with black bars so the final video is of constant size
def fit_to_resolution(frame, width, height):
    frame_height, frame_width = frame.shape[:2]

    if (
        frame_height < height or frame_width < width
    ):  # Frame is smaller, pad with black bars
        result = np.zeros((height, width, 3), dtype=frame.dtype)  # Create black canvas
        start_y = (height - frame_height) // 2
        start_x = (width - frame_width) // 2
        result[
            start_y : start_y + frame_height, start_x : start_x + frame_width
        ] = frame
    elif frame_height > height or frame_width > width:  # Frame is bigger, downsize
        result = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    else:  # Frame is already the correct size, return as is
        result = frame

    return result


if __name__ == "__main__":
    # frames, bounding_boxes, width, height = get_frames('starter_images/walking_vid.mp4')
    pass