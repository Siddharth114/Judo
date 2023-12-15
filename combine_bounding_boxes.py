from ultralytics import YOLO
import cv2
from PIL import Image
from itertools import combinations
import numpy as np

model = YOLO("yolov8n.pt")


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

    # return (
    #     max(x1, x2) - min(x1, x2) - minx_w(x1, w1, x2, w2) < x_val
    #     and max(y1, y2) - min(y1, y2) - miny_h(y1, h1, y2, h2) < y_val
    # )


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



    # x = min(box1[0], box2[0])
    # y = min(box1[1], box2[1])
    # w = max(box1[0] + box1[2], box2[0] + box2[2]) - x
    # h = max(box1[1] + box1[3], box2[1] + box2[3]) - y
    # return x, y, w, h


minx_w = lambda x1, w1, x2, w2: w1 if x1 <= x2 else w2
miny_h = lambda y1, h1, y2, h2: h1 if y1 <= y2 else h2


def draw_boxes(t, img, ind, coords_to_crop=[]):
    for i in t:
        # x0 = i[0] - i[2] / 2
        # x1 = i[0] + i[2] / 2
        # y0 = i[1] - i[3] / 2
        # y1 = i[1] + i[3] / 2
        # start_point = (int(x0), int(y0))
        # end_point = (int(x1), int(y1))

        (x1,y1,x2,y2,) = i

        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))

        img = cv2.rectangle(
            img, start_point, end_point, color=(0,0,255), thickness=5
        )


        # im_arr = cv2.rectangle(
        #     img, start_point, end_point, color=(0, 0, 255), thickness=5
        # )
        # img = im_arr

    if coords_to_crop:
        crop_and_add(coords_to_crop, img, ind)
    cv2.imwrite(f'starter_images/merged{ind}.jpg', img)
    fin_img = Image.fromarray(img[..., ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # fin_img.show()


def crop_and_add(coords, img, ind):
    xmin, ymin, xmax, ymax = coords
    xmin -= 50
    ymin-=50
    xmax+=50
    ymax+=50

    cropped_image = img[int(ymin):int(ymax), int(xmin):int(xmax)]

    cropped_aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0]

    height = img.shape[0]
    width = height * cropped_aspect_ratio
    
    cropped_image = cv2.resize(cropped_image, (int(width), int(height)), interpolation = cv2.INTER_AREA)
    
    horizontal_concat = np.concatenate((img, cropped_image), axis=1)

    cv2.imwrite(f'starter_images/original_and_cropped{ind}.jpg', horizontal_concat)



t = []
bounding_boxes = []
final_boxes = []
max_area = []

results = model.predict(
    ["starter_images/img1.jpg", 'starter_images/img2.jpg', 'starter_images/img3.jpg'], classes=[0], show_conf=False, show_labels=False
)
for i, r in enumerate(results):
    
    # bounding_boxes.extend(r.boxes.xywh.tolist())

    bounding_boxes.extend(r.boxes.xyxy.tolist())


    for j, k in combinations(r.boxes.xyxy.tolist(), r=2):
        if mergeable(j, k):
            t.append(list(merge(j, k)))
            try:
                bounding_boxes.remove(j)
                bounding_boxes.remove(k)
            except ValueError:
                pass

    # print(t+bounding_boxes)

    final_boxes = t+bounding_boxes

    max_area = max(final_boxes, key=lambda coord: (coord[2]-coord[0])*(coord[3]-coord[1]))

    draw_boxes(final_boxes, cv2.imread(f"starter_images/img{i+1}.jpg"), i+1, max_area)

    bounding_boxes=[]
    t=[]

    # im_arr = r.plot()
    # im = Image.fromarray(im_arr[..., ::-1])
    # im.show()
