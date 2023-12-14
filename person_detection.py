from ultralytics import YOLO
import cv2
from PIL import Image
from itertools import combinations
from numpy import asarray

def mergeable(box1, box2, x_val=100, y_val=100):
    (
        x1,
        y1,
        w1,
        h1,
    ) = box1
    (
        x2,
        y2,
        w2,
        h2,
    ) = box2
    return (
        max(x1, x2) - min(x1, x2) - minx_w(x1, w1, x2, w2) < x_val
        and max(y1, y2) - min(y1, y2) - miny_h(y1, h1, y2, h2) < y_val
    )


def merge(box1, box2):
    x = min(box1[0], box2[0])
    y = min(box1[1], box2[1])
    w = max(box1[0] + box1[2], box2[0] + box2[2]) - x
    h = max(box1[1] + box1[3], box2[1] + box2[3]) - y
    return x, y, w, h


minx_w = lambda x1, w1, x2, w2: w1 if x1 <= x2 else w2
miny_h = lambda y1, h1, y2, h2: h1 if y1 <= y2 else h2


def draw_boxes(t, img, ind=''):
    if not t:
        return img
    for i in t:
        x0 = i[0] - i[2] / 2
        x1 = i[0] + i[2] / 2
        y0 = i[1] - i[3] / 2
        y1 = i[1] + i[3] / 2
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))


        im_arr = cv2.rectangle(
            img, start_point, end_point, color=(0, 0, 255), thickness=5
        )
        img = im_arr

    # cv2.imwrite(f'starter_images/merged{ind}.jpg', img)
    # img = Image.fromarray(im_arr[..., ::-1])
    # img.show()
    return im_arr

def main():
    # model = YOLO('runs/detect/train/weights/best.pt') # engaged detection
    model = YOLO('yolov8n.pt') # person detection
    frames=[]
    

    cap = cv2.VideoCapture('starter_images/vid1.mp4')

    while cap.isOpened():
        bounding_boxes=[]
        t=[]
        success, frame = cap.read()
        if success:
            results = model.predict(frame, classes=[0], augment=False, show_conf=False, show_labels=False, conf=0.8)
            
            for r in results:
                bounding_boxes.extend(r.boxes.xywh.tolist())
                for j,k in combinations(r.boxes.xywh.tolist(), r=2):
                    if mergeable(j,k):
                        t.append(list(merge(j,k)))
                        try:
                            bounding_boxes.remove(j)
                            bounding_boxes.remove(k)
                        except ValueError:
                            pass
            
            annotated_frame = draw_boxes(t+bounding_boxes, frame)
            
            
            # annotated_frame = results[0].plot()
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
video_name = "starter_images/merged_output.mp4"
fps = 25
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_size = (960, 540)

writer = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

for frame in frames:
    writer.write(frame)

writer.release()