from ultralytics import YOLO
import cv2
from PIL import Image

model = YOLO('yolov8n.pt')

bounding_boxes = []

results = model.predict(['starter_images/img1.jpg', 'starter_images/img2.jpg', 'starter_images/img3.jpg'], classes=[0], show_conf=False, show_labels=False) 
for i,r in enumerate(results):
    print(r.boxes.xyxy)
    im_arr = r.plot()
    im = Image.fromarray(im_arr[..., ::-1])

    # im.show()