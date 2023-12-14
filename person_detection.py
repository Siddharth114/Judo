from ultralytics import YOLO
import cv2
from PIL import Image
from itertools import combinations
from numpy import asarray

def mergeable(box1, box2, x_val=100, y_val=100):
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

    intersection_check = not (
        (x3>x_val+x2 or x4+x_val<x1) or (y3>y2+y_val or y4+y_val<y1)
    )


    w1 = x2-x1
    w2 = x4-x3
    h1 = y2-y1
    h2 = y4-y3

    dimension_check = not ( (w1<.5*w2 and h1<.5*h2) or (w2<.5*w1 and h2<.5*h1))

    area1 = w1*h1
    area2 = w2*h2

    # area_check = not ( (area1<.5*area2) or (area2<.5*area1) )
    
    return intersection_check and dimension_check


def merge(box1, box2):
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

    x_min = min(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    x_max = max(x1, x2, x3, x4)
    y_max = max(y1, y2, y3, y4)

    return x_min, y_min, x_max, y_max


minx_w = lambda x1, w1, x2, w2: w1 if x1 <= x2 else w2
miny_h = lambda y1, h1, y2, h2: h1 if y1 <= y2 else h2


def draw_boxes(t, img, ind=''):
    if not t:
        return img
    for i in t:
        (x1,y1,x2,y2,) = i

        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))

        img = cv2.rectangle(
            img, start_point, end_point, color=(0,0,255), thickness=5
        )

    # cv2.imwrite(f'starter_images/merged{ind}.jpg', img)
    # fin_img = Image.fromarray(im_arr[..., ::-1])
    # fin_img.show()
    return img

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
                bounding_boxes.extend(r.boxes.xyxy.tolist())

                for j,k in combinations(r.boxes.xyxy.tolist(), r=2):
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