from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

def draw_boxes(frame, boxes):
    for i in boxes:
        xyxy, id = i
        x1, y1, x2, y2 = xyxy
        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))
        frame = cv2.rectangle(frame, start_point, end_point, color=(0, 0, 255), thickness=3)
        frame = cv2.putText(frame, str(int(id)), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def main():
    frames = []
    model = YOLO('yolov8n.pt')

    video_path = "starter_images/walking_vid.mp4"

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

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

            # bounding_boxes = results[0].boxes.xyxy.tolist()

            annotated_frame = draw_boxes(frame, bounding_boxes)

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
    return frames, fps


frames, fps = main()
video_name = "starter_images/people_tracking_output_with_ids_and_lines.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
height, width = frames[0].shape[0], frames[0].shape[1]

frame_size = (width, height)

writer = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

for frame in frames:
    writer.write(frame)

writer.release()