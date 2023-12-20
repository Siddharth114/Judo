from flask import Flask, render_template, request
import json
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

def decode_save(base64_str):
    data=base64_str.replace(' ','+')
    imgdata = base64.b64decode(data)
    filename = 'starter_images/img_from_site.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
            f.write(imgdata)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    print('test')
    output = request.get_json()
    result = output['value']
    decode_save(result)
    # print(result, type(result))
    return result

if __name__=='__main__':
    app.run(debug=True, port=8000)