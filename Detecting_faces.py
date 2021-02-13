from __future__ import print_function
import os
os.chdir(r"E:\Python coding\Projects\Attention")

import cv2
import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print(face_cascade)

img = cv2.imread('JenniferGroup.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(faces.shape)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

a = []
for i in range(0, faces.shape[0]):
    a.append(gray[faces[i][1]:faces[i][1]+faces[i][3],
                  faces[i][0]:faces[i][0]+faces[i][2]])
print(a)

plt.imshow(a[1],cmap=plt.get_cmap('gray'))

for k in range(0,faces.shape[0]):    
    print(a[k].shape)
    
import imageio
shape=28

img1=[]
img2=[]
for i in range(0,faces.shape[0]):    
    imageio.imwrite('face{}.jpg'.format(i), a[i])
    img1.append(cv2.cvtColor(cv2.imread('face{}.jpg'.format(i)), cv2.COLOR_BGR2GRAY))
    img2.append(cv2.resize(img1[i], (shape, shape)))

img2=np.array(img2)

for k in range(0,faces.shape[0]):    
    print(img2[k].shape)
    
