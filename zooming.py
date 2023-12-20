from ultralytics import YOLO
import cv2
from PIL import Image

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


model = YOLO('yolov8n.pt')
conf_thresh = 0.5
def main():
    model = YOLO('yolov8n.pt')
    frames=[]
    video_path = "starter_images/walking_vid.mp4"

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model.predict(frame, classes=[0], show_conf=False, show_labels=False)

            bounding_boxes = results[0].boxes.xyxy.tolist()

            # Visualize the results on the frame
            annotated_frame = draw_boxes(frame, bounding_boxes)

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

    return frames, bounding_boxes


frames, bounding_boxes = main()

video_name = "static/imgs/walking_vid_output.mp4"
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'avc1')
width, height = frames[0].shape[:2]

frame_size=(1280, 720)

print(frames[0].shape)

writer = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

for frame in frames:
    writer.write(frame)

writer.release()