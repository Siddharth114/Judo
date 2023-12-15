from ultralytics import YOLO
import cv2
from PIL import Image
from itertools import combinations
import numpy as np

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

    dimension_check = not ( (w1<.75*w2 and h1<.75*h2) or (w2<.75*w1 and h2<.75*h1))

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


def draw_boxes(t, img, coords_to_crop):
    if not t:
        img = crop_and_add(t, img)
        return img
    for i in t:
        (x1,y1,x2,y2,) = i

        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))

        img = cv2.rectangle(
            img, start_point, end_point, color=(0,0,255), thickness=5
        )
    fin_img = crop_and_add(coords_to_crop, img)
    # cv2.imwrite(f'starter_images/merged{ind}.jpg', img)
    # fin_img = Image.fromarray(im_arr[..., ::-1])
    # fin_img.show()
    return fin_img

def crop_and_add(coords, img):

    height, width, channels = img.shape

    black_rectangle = np.zeros((height, width, channels), dtype=img.dtype)

    canvas = np.concatenate((img, black_rectangle), axis=1)

    if not coords:
        return canvas
        # xmin, ymin, xmax, ymax = [0,0,500,img.shape[1]]
        # width = img.shape[1]
        # height = img.shape[0]
        # cropped_image = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        # horizontal_concat = np.concatenate((img, cropped_image), axis=1)
        # horizontal_concat = cv2.rectangle(horizontal_concat, (xmin+width, ymin), (xmax+width, ymax), color=(0,0,0), thickness=-1)


    else:
        xmin, ymin, xmax, ymax = coords
        xmin = max(0,xmin-50)
        ymin = max(0, ymin-50)
        xmax = min(img.shape[1],xmax+50)
        ymax = min(img.shape[0],ymax+50)

        x_overlay = int(width)
        y_overlay = 0
        cropped_image = img[int(ymin):int(ymax), int(xmin):int(xmax)]

        cropped_aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0]

        cropped_height = img.shape[0]
        cropped_width = height * cropped_aspect_ratio
        
        cropped_image = cv2.resize(cropped_image, (int(cropped_width), int(cropped_height)), interpolation = cv2.INTER_AREA)

        canvas[y_overlay:y_overlay+int(cropped_height), x_overlay:x_overlay+int(cropped_width)] = cropped_image

        return canvas




def main():
    # model = YOLO('runs/detect/train/weights/best.pt') # engaged detection
    model = YOLO('yolov8n.pt') # person detection
    frames=[]
    

    # cap = cv2.VideoCapture('starter_images/vid1.mp4') #longer video
    cap = cv2.VideoCapture('starter_images/vid2.mp4') #shorter video

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
            
            final_boxes = t+bounding_boxes

            if final_boxes:
                max_area = max(final_boxes, key=lambda coord: (coord[2]-coord[0])*(coord[3]-coord[1]))
            else:
                max_area = []


            annotated_frame = draw_boxes(t+bounding_boxes, frame, max_area)
            
            
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

# print(frames[0].shape)



# writing the annotated frames to a video

video_name = "starter_images/merged_cropped_output.mp4"
fps = 25
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_size = (2560, 720)

writer = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

for frame in frames:
    writer.write(frame)

writer.release()