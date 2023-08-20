import numpy as np
import cv2
import io
import os
from PIL import Image
import base64
from flask import Flask, render_template, Response, request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from importlib import import_module
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.models import load_model
import pandas as pd
# from openpyxl import load_workbook

app = Flask(__name__,template_folder='./templates/',static_folder='./static/')

if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera


## Loading Face Detection feature

face_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

WHITE = [255, 255, 255]


## Bounding Box around face in a image
def draw_box(Image, x, y, w, h):
    cv2.line(Image, (x, y), (x + int(w / 5), y), WHITE, 2)
    cv2.line(Image, (x + int((w / 5) * 4), y), (x + w, y), WHITE, 2)
    cv2.line(Image, (x, y), (x, y + int(h / 5)), WHITE, 2)
    cv2.line(Image, (x + w, y), (x + w, y + int(h / 5)), WHITE, 2)
    cv2.line(Image, (x, (y + int(h / 5 * 4))), (x, y + h), WHITE, 2)
    cv2.line(Image, (x, (y + h)), (x + int(w / 5), y + h), WHITE, 2)
    cv2.line(Image, (x + int((w / 5) * 4), y + h), (x + w, y + h), WHITE, 2)
    cv2.line(Image, (x + w, (y + int(h / 5 * 4))), (x + w, y + h), WHITE, 2)

emotion_labels = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
            'Disquietment', 'Doubt', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue',
            'Fear', 'Happiness','Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise',
            'Sympathy', 'Yearning']

# Plot Function which returns the Plot Image
def plot(emotion_label, confidence_score , pred):
    # Define a color map for the emotions
    cmap = plt.get_cmap('tab20b')

    # Predict the emotion label of the image
    # emotion_label, confidence_score , pred = predict_emotion(image_path)

    # Plot a bar graph with the predicted emotions and their corresponding confidence scores
    plt.tight_layout()
    fig, ax = plt.subplots(figsize=(28, 8))
    bars = ax.bar(emotion_labels, pred, color=cmap(range(len(emotion_labels))))
    plt.xticks(rotation=90)
    plt.xlabel('Emotion')
    plt.ylabel('Confidence Score')
    plt.title(f'Predicted Emotion: {emotion_label} (Confidence: {confidence_score:.2f})')

    # Add a legend to the bar graph
    handles = [mpatches.Patch(color=cmap(i), label=emotion_labels[i]) for i in range(len(emotion_labels))]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    buf=io.BytesIO()
    plt.savefig(buf,format='png')
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode('utf-8')
    
    return graph

## Returns the index for sorted elements
def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s


##For Predicting Images and showing the Emotion Label
def predict_emotion(image_path,model):
    # Load the image and preprocess it
    # model = load_model("Alexnet_Emotic_Adam.h5")
    top3=[]
    emotion_label=[]
    # img = load_img(image_path, target_size=(224, 224))

    img = img_to_array(image_path)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet.preprocess_input(img)

    # Make a prediction using the loaded model
    pred = model.predict(img)

    # Get the predicted emotion label and confidence score
    Pred = pred[0]
    top3.extend(sort_index(pred[0])[:3])
    # print(top3)
    for i in range(3):
        emotion_label.append(emotion_labels[top3[i]])
    confidence_score = pred[0][top3[0]]
    return emotion_label, confidence_score, Pred


## Home Route
@app.route('/')
def index():
    cv2.destroyAllWindows()
    return render_template('Home_index.html')

## Camera Feed Route
@app.route('/camera_feed')
def camera_feed():
    """Video streaming home page."""
    return render_template('result.html')

def gen(camera):
    """Video streaming generator function."""
    yield b'--frame\r\n'
    while True:
        frame = camera.get_frame()
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

## For Prediction page
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    cv2.destroyAllWindows()
    model = load_model("Alexnet_Emotic_Adam.h5")
    if request.method == 'POST':
            if request.form.get('Analyze') == 'Analyze':
                input_data = list(request.form.values())
                print(input_data)
                model = load_model("Alexnet_Emotic_Adam.h5")
                emotion_label=[]
                # input_data[0]='Alexnet'
                if input_data[0] == 'Alexnet_Emotic_Adam_TL':
                    model = load_model("static/Models/Models_TL/Alexnet_Emotic_Adam (1).h5")
                elif input_data[0] == 'Alexnet_Emotic_RMSprop_TL':
                    model = load_model("static/Models/Models_TL/ALEXNET_Emotic_RMSprop (1).h5")
                else:
                    model = load_model("static/Models/Models_TL/ALEXNET_Emotic_SGD (1).h5")

                camera = cv2.VideoCapture(0)
                success, frame = camera.read()  # read the camera frame
                frame = cv2.resize(frame, (224, 224))
                r, jpg = cv2.imencode('.jpg', frame)
                # print("successful")
                
                emotion_label, confidence_score, pred = predict_emotion(frame,model)
                # print(emotion_label)
                graph=plot(emotion_label[0], confidence_score, pred)
                mylist = [emotion_label[0], confidence_score]
                output=mylist[1]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(frame,1.1,7)
                for (x, y, w, h) in faces:
                    gray_face = cv2.resize((gray[y:y + h, x:x + w]), (110, 110))
                    eyes = eye_cascade.detectMultiScale(gray_face)
                    for (ex, ey, ew, eh) in eyes:
                        draw_box(gray, x, y, w, h)
                r, jpg = cv2.imencode('.jpg', gray)
                print("successful")
                img= jpg.tobytes()
                img = base64.b64encode(img).decode('utf-8')
                # print("successful")
                return render_template('prediction.html',Image=img,emotion_label=emotion_label,pred_model=input_data[0],graph=graph,prediction_text=" The {} is having a confidence score = {}".format(mylist[0],output),Models=[{'Model':'Alexnet_Emotic_Adam_TL'},{'Model':'Alexnet_Emotic_RMSprop_TL'},{'Model':'Alexnet_Emotic_SGD_TL'}])
            
            elif request.form.get('Upload') == 'Upload':
                # print("successful post 2")
                emotion_label=[]
                
                input_data = list(request.form.values())
                print(input_data)
                # input_data[0]='Alexnet'
                if input_data[0] == 'Alexnet_Emotic_Adam_TL':
                    model = load_model("static/Models/Models_TL/Alexnet_Emotic_Adam (1).h5")
                elif input_data[0] == 'Alexnet_Emotic_RMSprop_TL':
                    model = load_model("static/Models/Models_TL/ALEXNET_Emotic_RMSprop (1).h5")
                else:
                    model = load_model("static/Models/Models_TL/ALEXNET_Emotic_SGD (1).h5")
                file = request.files['file']
                img=Image.open(file)
                img= img.resize((224,224))
                img=img.convert('RGB') 
                # img = load_img(file, target_size=(224, 224))                
                emotion_label, confidence_score, pred = predict_emotion(img,model)
                # print(emotion_label)
                graph=plot(emotion_label[0], confidence_score, pred)
                mylist = [emotion_label[0], confidence_score]
                output=mylist[1]
                cv_image = np.array(img) 
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(cv_image,1.1,7)
                for (x, y, w, h) in faces:
                    gray_face = cv2.resize((gray[y:y + h, x:x + w]), (110, 110))
                    eyes = eye_cascade.detectMultiScale(gray_face)
                    for (ex, ey, ew, eh) in eyes:
                        draw_box(gray, x, y, w, h)
                r, jpg = cv2.imencode('.jpg', gray)
                file= jpg.tobytes()
                file = base64.b64encode(file).decode('utf-8')
                # print("successful")
                return render_template('prediction.html',file_img=file,emotion_label=emotion_label,pred_model=input_data[0],graph=graph,prediction_text=" The {} is having a confidence score = {}".format(mylist[0],output),Models=[{'Model':'Alexnet_Emotic_Adam_TL'},{'Model':'Alexnet_Emotic_RMSprop_TL'},{'Model':'Alexnet_Emotic_SGD_TL'}])

            else:
                # pass # unknown
                return render_template("prediction.html",Models=[{'Model':'Alexnet_Emotic_Adam_TL'},{'Model':'Alexnet_Emotic_RMSprop_TL'},{'Model':'Alexnet_Emotic_SGD_TL'}])
    elif request.method == 'GET':
            print("successful Get")
            
            return render_template('prediction.html',Models=[{'Model':'Alexnet_Emotic_Adam_TL'},{'Model':'Alexnet_Emotic_RMSprop_TL'},{'Model':'Alexnet_Emotic_SGD_TL'}])
    
@app.route('/presentation')
def presentation():
    return render_template('presentation.html')

@app.route("/Models")
def model():
    return render_template('models.html')

@app.route("/docs")
def docs():
    return render_template('docs.html')

if __name__=='__main__':
    app.run(debug=True)