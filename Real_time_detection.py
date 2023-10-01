from keras.models import model_from_json
import numpy as np
import cv2

json_file = open("emotionsdetectormodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotionsdetectormodel.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def ef(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

webcam = cv2.VideoCapture(0)

labels = {0:'angry', 1:'disgusted', 2:'fearful', 3:'happy', 4:'neutral', 5:'sad', 6:'surprised'}

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    try:
        for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,222,0),2)
            image = cv2.resize(image, (48,48))
            img = ef(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,222,255))

        cv2.imshow("Output", im)
        cv2.waitKey(27)
    except cv2.error:
        pass
