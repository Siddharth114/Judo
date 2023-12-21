from flask import Flask, render_template, request, jsonify
import json
import base64
from io import BytesIO
from PIL import Image
import zooming
import numpy as np
import cv2
import pickle

app = Flask(__name__)

# def decode_save(base64_str):
#     data=base64_str.replace(' ','+')
#     imgdata = base64.b64decode(data)

#     img_array = cv2.imdecode(np.frombuffer(imgdata, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
#     return img_array

    # filename = 'starter_images/img_from_site.jpg'
    # with open(filename, 'wb') as f:       # write image to file
    #         f.write(imgdata)            

def numpy_array_to_base64(frame, format="jpeg"):
    success, encoded_bytes = cv2.imencode(".jpg", frame)
    base64_string = base64.b64encode(encoded_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_string}"
    return data_uri        

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    output = request.get_json()
    mouse_x = output['mouseX']
    mouse_y = output['mouseY']
    vid_width = output['vidWidth']
    vid_height = output['vidHeight']
    current_frame = output['frameNumber']
    
    stopped_frame = frames[current_frame]
    boxes = bounding_boxes[current_frame]
    
    new_x, new_y = zooming.translate_coords(mouse_x, mouse_y, vid_width, vid_height, original_width, original_height)

    cropped_image = zooming.cropped_img(stopped_frame, boxes, new_x, new_y)

    processed_image = {
        'image': numpy_array_to_base64(cropped_image),
        'msg': 'did not click inside a box'
    }

    return jsonify(processed_image)

if __name__=='__main__':
    video_path = 'starter_images/walking_vid.mp4'
    frames, bounding_boxes, original_width, original_height, fps = zooming.get_frames(video_path)  #used to run the model and get boxes for the video    
    output_video_path = '/Users/siddharth/Code/Python/Judo/static/imgs/walking_vid_output.mp4'
    # zooming.write_video(frames, output_video_path, original_width, original_height, fps)
    app.run(port=8000)