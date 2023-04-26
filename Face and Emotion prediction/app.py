from flask import Flask, render_template, Response, request
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import cv2

# app=Flask(__name__)
app = Flask(__name__,template_folder='./templates/',static_folder='./css/')

camera = cv2.VideoCapture(0)



## Loading Models

face_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

WHITE = [255, 255, 255]

emotion_labels = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
            'Disquietment', 'Doubt', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue',
            'Fear', 'Happiness','Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise',
            'Sympathy', 'Yearning']

## Default model
model = load_model("Alexnet_Emotic_Adam.h5")


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

## For continusly watching the video frame
def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            faces=detector.detectMultiScale(frame,1.1,7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             #Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

##For Predicting Images and showing the Emotion Label

def predict_emotion(image_path):
    # Load the image and preprocess it
    # img = load_img(image_path, target_size=(224, 224))

    img = img_to_array(image_path)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet.preprocess_input(img)

    # Make a prediction using the loaded model
    pred = model.predict(img)

    # Get the predicted emotion label and confidence score
    emotion_index = np.argmax(pred)
    emotion_label = emotion_labels[emotion_index]
    confidence_score = pred[0][emotion_index]
    Pred = pred[0]

    return emotion_label, confidence_score, Pred



@app.route('/')
def index():
    return render_template('index.html',Models=[{'Model':'Alexnet'},{'Model':'VGG19'},{'Model':'DenseNet'}])
    # return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    input_data = list(request.form.values())
    print(input_data[0])
    if input_data[0] == 'AlexNet':
        model = load_model("Alexnet_Emotic_Adam.h5")
    elif input_data[0] == 'VGG19':
        model = load_model("Alexnet_Emotic_Adam.h5")
    else:
        model = load_model("Alexnet_Emotic_Adam.h5")


    if request.method == 'POST':
            print("Post running")
            if request.form.get('Analyze') == 'Analyze':
                # print("successful post 1")
                camera = cv2.VideoCapture(0)
                success, frame = camera.read()  # read the camera frame
                frame = cv2.resize(frame, (224, 224))

                emotion_label, confidence_score, pred = predict_emotion(frame)
                mylist = [emotion_label, confidence_score]
                output=mylist[1]
                # print("successful")
                return render_template('index.html', mylist=mylist,emotion_label=mylist[0],prediction_text=" The predicted emotion label is having a confidence score = {}".format(output),Models=[{'Model':'Alexnet'},{'Model':'VGG19'},{'Model':'DenseNet'}])

            elif  request.form.get('Watch') == 'Watch':
                # print("successful post 2")
                # pass # do something else
                camera = cv2.VideoCapture(0)
                success, frame = camera.read()  # read the camera frame
                # frame = cv2.resize(frame, (224, 224))
                # image=frame
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(frame,1.1,7)
                for (x, y, w, h) in faces:
                    gray_face = cv2.resize((gray[y:y + h, x:x + w]), (110, 110))
                    eyes = eye_cascade.detectMultiScale(gray_face)
                    for (ex, ey, ew, eh) in eyes:
                        draw_box(gray, x, y, w, h)

                r, jpg = cv2.imencode('.jpg', gray)
                # print("successful")
                return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')
            else:
                # pass # unknown
                return render_template("index.html")
    elif request.method == 'GET':
            # return render_template("index.html")
            # print("No Post Back Call")
            print("successful Get")
            
            return render_template('index.html',Models=[{'Model':'Alexnet'},{'Model':'VGG19'},{'Model':'DenseNet'}])
if __name__=='__main__':
    app.run(debug=True)