from ultralytics import YOLO
import cv2
from PIL import Image

model = YOLO('yolov8n.pt')
conf_thresh = 0.5


cap = cv2.VideoCapture('test_images/vid1.mp4')

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.predict(frame, classes=[0], augment=False, show_conf=False, show_labels=False)
        annotated_frame = results[0].plot()

        cv2.imshow("Person Detection", annotated_frame)

        if cv2.waitKey(1) and 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()


# results = model.predict(['images/img1.jpg', 'images/img2.jpg', 'images/img3.jpg'], classes=[0], show_conf=False, show_labels=False) 
# for i,r in enumerate(results):
#     im_arr = r.plot()
#     im = Image.fromarray(im_arr[..., ::-1])
#     # im.show()
#     im.save(f'results{i}.jpg')
