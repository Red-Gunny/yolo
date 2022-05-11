import io
import torch
from flask import Flask, request, jsonify
from PIL import Image

from werkzeug.utils import secure_filename

app = Flask(__name__)

def make_list(detection) :
    all_names = detection.names
    liquors = []
    liquor = {}
    for predict_result in detection.pred :          # tensor 객체의 리스트 요소 각각에 대하야
        predict_result_list=predict_result.numpy().tolist() # 텐서 객체를 리스트로 바꿈    (prediction 결과 1개)
        index = 0
        for val in predict_result_list :
            liquor['index'] = index
            liquor['name'] = all_names[int(val[5])]
            liquor['accuracy'] = round(val[4],2)
            index+=1
            liquors.append(liquor)
            liquor={}
    return liquors

def make_dictionary(liquor_list) :
    dictionary={}
    dictionary['size'] = len(liquor_list)
    dictionary['liquors']=liquor_list
    return dictionary

@app.route('/yolo', methods=['POST'])
def detect_image():
    if request.method == 'POST':
        f = request.files['file']
        f_bytes = f.read()
        img=Image.open(io.BytesIO(f_bytes))
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weight/last.pt')
        result = model(img, size=640)
        liquors = make_list(result)
        dictionary_result = make_dictionary(liquors)
        return jsonify(dictionary_result)

@app.route('/test', methods=['POST'])
def image_test():
   if request.method == 'POST':
       f = request.files['file']
       f.save(secure_filename(f.filename))
       liquor_image = open("./test_image/image_1.jpg", 'rb')
       f_bytes = liquor_image.read()
       img = Image.open(io.BytesIO(f_bytes))
       model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weight/last.pt')
       result = model(img, size=640)
       liquors = make_list(result)
       dictionary_result = make_dictionary(liquors)
       liquor_image.close()
       return jsonify(dictionary_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=40000)
