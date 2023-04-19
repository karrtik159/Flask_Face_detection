from flask import Flask, render_template, Response, request
# /  g, send_file, Blueprint
# from io import BytesIO
# from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import cv2

WHITE = [255, 255, 255]
app=Flask(__name__)
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
model = load_model(r"C:\Users\ASUS\Downloads\Alexnet_Emotic_Adam.h5")
emotion_labels = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
            'Disquietment', 'Doubt', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue',
            'Fear', 'Happiness','Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise',
            'Sympathy', 'Yearning']

# @app.before_request
# def before_request():
#     g.img = None
#     g.user = None
def draw_box(Image, x, y, w, h):
    cv2.line(Image, (x, y), (x + int(w / 5), y), WHITE, 2)
    cv2.line(Image, (x + int((w / 5) * 4), y), (x + w, y), WHITE, 2)
    cv2.line(Image, (x, y), (x, y + int(h / 5)), WHITE, 2)
    cv2.line(Image, (x + w, y), (x + w, y + int(h / 5)), WHITE, 2)
    cv2.line(Image, (x, (y + int(h / 5 * 4))), (x, y + h), WHITE, 2)
    cv2.line(Image, (x, (y + h)), (x + int(w / 5), y + h), WHITE, 2)
    cv2.line(Image, (x + int((w / 5) * 4), y + h), (x + w, y + h), WHITE, 2)
    cv2.line(Image, (x + w, (y + int(h / 5 * 4))), (x + w, y + h), WHITE, 2)

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
def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        # frame = cv2.resize(frame, (224, 224))
        # image=frame
        if not success:
            break
        else:
            detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces=detector.detectMultiScale(frame,1.1,7)

            #Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
            if request.form.get('Encrypt') == 'Encrypt':
                camera = cv2.VideoCapture(0)
                success, frame = camera.read()  # read the camera frame
                frame = cv2.resize(frame, (224, 224))

                emotion_label, confidence_score, pred = predict_emotion(frame)
                mylist = [emotion_label, confidence_score]
                return render_template('index.html', mylist=mylist)

            elif  request.form.get('Decrypt') == 'Decrypt':
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

                return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')
            else:
                # pass # unknown
                return render_template("index.html")
    elif request.method == 'GET':
            # return render_template("index.html")
            print("No Post Back Call")
    return render_template("index.html")
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)
