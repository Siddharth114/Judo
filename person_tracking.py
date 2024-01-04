from ultralytics import YOLO
import cv2


def main():
    frames = []
    model = YOLO('yolov8n.pt')

    video_path = "starter_images/walking_vid.mp4"

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True, verbose=True)

            annotated_frame = results[0].plot()
            frames.append(annotated_frame)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    return frames


frames = main()
video_name = "starter_images/people_tracking_output.mp4"
fps = 20
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
height, width = frames[0].shape[0], frames[0].shape[1]

frame_size = (width, height)

writer = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

for frame in frames:
    writer.write(frame)

writer.release()