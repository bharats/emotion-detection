import cv2
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

total = pd.read_csv('icml_face_data.csv')
total = total.drop(columns = [' Usage'])

total_y = total['emotion']
total_x = total[' pixels'].str.split(' ',expand=True).astype(dtype = 'uint8')

x_train, x_test, y_train, y_test  = train_test_split(total_x, total_y, test_size=0.25, random_state=42)








sc = MinMaxScaler()
x_train_sc = sc.fit_transform(x_train)
save_sc = sc.fit(x_train)

pca = PCA(n_components=255).fit(x_train_sc)
pca_save = PCA(n_components=255)
pca_s = pca_save.fit(x_train_sc)


x_train_pca = pca.transform(x_train_sc)

lrc = LogisticRegression(C=0.01, penalty='l2', solver='saga', tol=0.0001, max_iter=800)

pipe = Pipeline([('pca', pca), ('logistic', lrc)])
pipe.fit(x_train_pca, y_train)









#--------------------------------------------------------------------------------------

cap = cv2.VideoCapture(1) #default is normally 0 but my webcam is 1
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = cascade.detectMultiScale(gray, 1.3,5)

    if(len(detections)>0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        face = Image.fromarray(gray[y-10:y+h+10,x-10:x+w+10])
        face = face.resize((48, 48), Image.ANTIALIAS)
        face = np.resize(np.array(face), (1,2304))

        face_sc = save_sc.transform(face)
        face_pca = pca_s.transform(face_sc)
        pred = pipe.predict(face_pca)

        emotion = {0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Suprised", 6:"Neutral"}


        cv2.putText(frame,str(emotion.get(pred[0])),(x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_4)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
