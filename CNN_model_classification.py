# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:21:13 2021

@author: Acer
"""
import cv2
import numpy
import matplotlib.pyplot as plt
import os
import random
import pickle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Activation, Flatten 

datadir = "E:/Python coding/Classification/Natural_Images"
categories = ["airplane", "car", "dog", "cat", "flower", "fruit", "motorbike", "person"]

train_data = []

def create_train_data():
    for category in categories:
        path = os.path.join(datadir, category)
        class_index = categories.index(category)
        for img in os.listdir(path):
            arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(arr, (80, 80))
            train_data.append([resized_arr, class_index])
       
create_train_data()
    
print(len(train_data))

random.shuffle(train_data)
    
input_data = []
class_label = []

for features, label in train_data:
    input_data.append(features)
    class_label.append(label)
    
ip_features = numpy.array(input_data).reshape(-1, 80, 80, 1)

# pickle_outX = open("features.pickle", "wb")
# pickle.dump(ip_features, pickle_outX)
# pickle_outX.close

# pickle_outy = open("class_labels.pickle", "wb")
# pickle.dump(class_label, pickle_outy)
# pickle_outy.close


# features = pickle.load(open("features.pickle", "rb"))
# class_labels = pickle.load(open("class_labels.pickle", "rb"))

features = ip_features/255.0
onehotencoder = OneHotEncoder()

class_array = numpy.asarray(class_label).reshape(-1, 1)
enc_class = onehotencoder.fit_transform(class_array).toarray()
print(enc_class.shape)

model = Sequential()

model.add(Conv2D(128, (5,5), input_shape = features.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (5,5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(8))
model.add(Activation("sigmoid"))

model.compile(loss="categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

model.fit(features, enc_class,
          batch_size = 32,
          validation_split = 0.2,
          epochs = 100)

