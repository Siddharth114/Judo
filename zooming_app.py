from flask import Flask, render_template, request
import json
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

def decode_save(base64_str):
    data=base64_str.replace(' ','+')
    imgdata = base64.b64decode(data)
    filename = 'starter_images/img_from_site.jpg'
    # with open(filename, 'wb') as f:       # write image to file
    #         f.write(imgdata)                          

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    output = request.get_json()
    frame_data = output['img']
    mouse_x = output['mouseX']
    mouse_y = output['mouseY']
    vid_width = output['vidWidth']
    vid_height = output['vidHeight']
    decode_save(frame_data)
    return frame_data

if __name__=='__main__':
    app.run(debug=True, port=8000)