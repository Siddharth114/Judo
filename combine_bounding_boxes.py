from ultralytics import YOLO
import cv2
from PIL import Image
from itertools import combinations
from numpy import asarray

model = YOLO("yolov8n.pt")


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


def draw_boxes(t, img, ind):
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

    cv2.imwrite(f'starter_images/combined{ind}.jpg', img)
    img = Image.fromarray(im_arr[..., ::-1])
    # img.show()


t = []
bounding_boxes = []


results = model.predict(
    ["starter_images/img1.jpg", 'starter_images/img2.jpg', 'starter_images/img3.jpg'], classes=[0], show_conf=False, show_labels=False
)
for i, r in enumerate(results):
    bounding_boxes.extend(r.boxes.xywh.tolist())
    for j, k in combinations(r.boxes.xywh.tolist(), r=2):
        # print('j,k= ',j,k)
        if mergeable(j, k):
            t.append(list(merge(j, k)))
            try:
                bounding_boxes.remove(j)
                bounding_boxes.remove(k)
            except ValueError:
                pass

    # print(t+bounding_boxes)

    draw_boxes(t + bounding_boxes, cv2.imread(f"starter_images/img{i+1}.jpg"), i+1)

    bounding_boxes=[]
    t=[]

    # im_arr = r.plot()
    # im = Image.fromarray(im_arr[..., ::-1])
    # im.show()
