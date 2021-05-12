import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt

total = pd.read_csv('icml_face_data.csv')
total = total.drop(columns = [' Usage'])

total_y = total['emotion']
total_x = total[' pixels'].str.split(' ',expand=True).astype(dtype = 'uint8')

x_train, x_test, y_train, y_test  = train_test_split(total_x, total_y, test_size=0.25, random_state=42)

sc = StandardScaler()

x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

pca = PCA(n_components=100)

x_train_pca = pca.fit_transform(x_train_sc)
x_test_pca = pca.transform(x_test_sc)

explained_variance = pca.explained_variance_ratio_



print(explained_variance)
print()
