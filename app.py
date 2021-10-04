import os
import sys
import cv2
import pprint
import numpy as np

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil
from preprocess import process_test
from preprocess import process

# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#model = MobileNetV2(weights='imagenet')

#print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models\ImageEmptionClassifierModel'

# Load your own trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

row = 80
cox = 80
# def process(imageName):
#     output = None
#     #check the format of image
#     #if imageName.endswith('jpg') or imageName.endswith('png') or imageName.endswith('jpeg'):
        
#         # read the image in grayscale
#     #grayscaleImage = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
#     image = cv2.imdecode(np.fromstring(imageName, np.uint8), cv2.IMREAD_GRAYSCALE)/ 255

#         # resize the image 28 * 28
#     output = cv2.resize(grayscaleImage, (80, 80))

#     return output
        

    #return the resul
    
        

    #return the result
    # return output
# TODO: we may need to assign output appropriately labelled here as output would be integers
def category_output(prediction):
  emotion_types = ["Damilola", "Damola", "Fatimah", "David", "Rasheed", "Oluleye"] 
  result = emotion_types[np.argmax(prediction)]
  return result

  
# def model_predict(img, model):
#     #img = img.resize((80, 80))

#     # Preprocessing the image
#     #x = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#     #x = np.true_divide(x, 255)
#     #x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x= img
#     processed_img = np.array(x)
#     feature_img = np.array(x).reshape(-1, 80, 80, 1)

#     preds = model.predict(feature_img)
#     return preds

# def model_predict(img, model):
#     img = img.resize((80, 80))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     #x = np.expand_dims(x, axis=0)
#     x = np.array(x).reshape(-1, 80, 80, 1)
#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     #x = preprocess_input(x, mode='tf')

#     preds = model.predict(x)
#     return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        img.save("./uploads/image.jpg")
        pat = './uploads/image.jpg'
        img = process(pat)
        #img = model.predict()
        processed_img = image.img_to_array(img)
        feature_img = np.array(processed_img).reshape(-1, 80, 80, 1)
        output = model.predict(feature_img)

        output_class = category_output(output)
        #print('Image: ',img, '\n') 
        return jsonify(result=output_class)
    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
