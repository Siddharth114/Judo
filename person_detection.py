from ultralytics import YOLO
import cv2

def main():
    model = YOLO('runs/detect/train/weights/best.pt') # engaged detection
    # model = YOLO('yolov8n.pt') # person detection
    frames=[]

    cap = cv2.VideoCapture('starter_images/vid1.mp4')

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.predict(frame, classes=[0], augment=False, show_conf=False, show_labels=False, conf=0.8)
            annotated_frame = results[0].plot()
            frames.append(annotated_frame)

            cv2.imshow("Person Detection", annotated_frame)

            if cv2.waitKey(1) and 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    return frames

frames = main()

# writing the annotated frames to a video
video_name = "output.mp4"
fps = 25
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_size = (960, 540)

writer = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

for frame in frames:
    writer.write(frame)

writer.release()