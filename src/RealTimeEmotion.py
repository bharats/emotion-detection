import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tkinter import *
from tkinter.ttk import *
from PIL import Image
import pickle


decomposition = " "
classifier = " "

#---------------------------------------------------------------------------

options1 = [
    "Decomposition",
    "PCA",
    "NMF",
    "Kernel PCA"
]
options2 = [
    "Classifier",
    "Logistic Regression",
    "Linear SVC",
    "Multilater Perceptron",
    "Random Forrest",
    "K-Nearest Neighbor"
]

def choose():
    global decomposition
    global classifier
    decomposition = clicked1.get()
    classifier = clicked2.get()
    root.destroy()

root = Tk()
root.geometry("300x100")

clicked1 = StringVar(root)
clicked1.set("Decomposition")

clicked2 = StringVar(root)
clicked2.set("Classifier")

drop1 = OptionMenu(root, clicked1, *options1)
drop1.pack()

drop2 = OptionMenu(root, clicked2, *options2)
drop2.pack()

button = Button(root, text="Start", command = choose).pack()

root.mainloop()


#----------------------------------------------------------------------------------

x_train_file = open('../data/processed/data_models/x_train.pkl', 'rb')
x_train = pickle.load(x_train_file)

x_sc_train = open('../data/processed/data_models/x_train_sc.pkl', 'rb')
x_train_sc =pickle.load(x_sc_train)

sc = MinMaxScaler()
save_sc = sc.fit(x_train)


if(decomposition =="PCA"):
    # pca
    text = 'pca'
    pca_file = open('../data/processed/data_models/x_train_pca.pkl', 'rb')
    x_train_m = pickle.load(pca_file)

    dec_file = open('../data/processed/data_models/pca.pkl', 'rb')
    dec = pickle.load(dec_file)

elif(decomposition == "NMF"):
    # nmf
    text = 'nmf'
    nmf_file = open('../data/processed/data_models/x_train_nmf.pkl', 'rb')
    x_train_m = pickle.load(nmf_file)

    dec_file = open('../data/processed/data_models/nmf.pkl', 'rb')
    dec = pickle.load(dec_file)

else:
    # kernel pca
    text = 'kpca'
    kpca_file = open('../data/processed/data_models/x_train_kpca.pkl', 'rb')
    x_train_m = pickle.load(kpca_file)

    dec_file = open('../data/processed/data_models/kpca.pkl', 'rb')
    dec = pickle.load(dec_file)




if(classifier == "Logistic Regression"):
    # logistic regression
    dec_class_file = open('../data/processed/data_models/lrc_'+text+'.pkl', 'rb')
    classif =pickle.load(dec_class_file)

elif(classifier == "Linear SVC"):
    # support vector machine classifier
    dec_class_file = open('../data/processed/data_models/scv_' + text + '.pkl', 'rb')
    classif = pickle.load(dec_class_file)

elif (classifier == "Multilater Perceptron"):
    # multilater perceptron
    dec_class_file = open('../data/processed/data_models/mlp_' + text + '.pkl', 'rb')
    classif = pickle.load(dec_class_file)

elif (classifier == "Random Forrest"):
    # random forrest
    dec_class_file = open('../data/processed/data_models/rfc_' + text + '.pkl', 'rb')
    classif = pickle.load(dec_class_file)

else:
    # k-nearest neighbor
    dec_class_file = open('../data/processed/data_models/knc_' + text + '.pkl', 'rb')
    classif = pickle.load(dec_class_file)


#----------------------------------------------------------------------------------

cap = cv2.VideoCapture(1) #default is normally 0 but my webcam is 1
cascade = cv2.CascadeClassifier('../data/raw/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = cascade.detectMultiScale(gray, 1.3,5)

    if(len(detections)>0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        face = Image.fromarray(gray[y:y+h,x:x+w])
        face = face.resize((48, 48), Image.ANTIALIAS)
        face = np.resize(np.array(face), (1,2304))

        face_sc = save_sc.transform(face)
        face_dec = dec.transform(face_sc)
        pred = classif.predict(face_dec)

        emotion = {0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Suprised", 6:"Neutral"}

        cv2.putText(frame,str(emotion.get(pred[0])),(x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_4)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
