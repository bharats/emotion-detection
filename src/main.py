import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import matplotlib.pyplot as plt

total = pd.read_csv('icml_face_data.csv')
total = total.drop(columns = [' Usage'])

total_y = total['emotion']
total_x = total[' pixels'].str.split(' ',expand=True).astype(dtype = 'uint8')

x_train, x_test, y_train, y_test  = train_test_split(total_x, total_y, test_size=0.25, random_state=42)

sc = MinMaxScaler()

x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

pca = PCA(n_components=100).fit(x_train_sc)

x_train_pca = pca.transform(x_train_sc)
x_test_pca = pca.transform(x_test_sc)

explained_variance = pca.explained_variance_ratio_

lrc = LogisticRegression(C=1000, tol=0.0001, max_iter=10000)

pipe = Pipeline([('pca', pca), ('logistic', lrc)])
pipe.fit(x_train_pca, y_train)
predictions = pipe.predict(x_test_pca)

print(confusion_matrix(y_test, lrc.predict(x_test_pca)))
print(classification_report(y_test, lrc.predict(x_test_pca)))

