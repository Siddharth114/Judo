from flask import Flask, render_template, request
import json
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

def decode_save(base64_str):
    binary_data = base64.b64decode(base64_str)
    image_buffer = BytesIO(binary_data)
    image = Image.open(image_buffer)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    print('test')
    output = request.get_json()
    result = output['value']
    # decode_save(result)
    # print(result, type(result))
    return result

if __name__=='__main__':
    app.run(debug=True)