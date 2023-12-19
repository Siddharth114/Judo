from flask import Flask, render_template, request
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    print('test')
    output = request.get_json()
    print(output, type(output))
    result = output['value']
    print(result, type(result))
    return result

if __name__=='__main__':
    app.run(debug=True)