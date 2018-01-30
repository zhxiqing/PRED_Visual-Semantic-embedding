import json
import keras
import numpy as np
import random
import cv2
import outil
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

preffix = './valCaptions/'
suffix = '.json'
classes = ['airplane','apple','banana',
            'backpack','stop sign','bed',
            'cup','bus','horse']
savePreffix = './VectorJson/Test/'
labelToClass = {}
data = []
imageName = []
images = []
captions = []
labels = []
modelPath = './ImageEmbedding.h5'
imagePath = '/media/zxq/zxq/COCO/Valresized/'

for m_class in classes:
    with open(preffix+m_class+suffix) as f:
        temp = json.load(f)
    labelToClass[temp[0]['label']] = m_class
    data += temp
    f.close()
print(len(data))
for i in range(0,len(data)):
    imageName.append(data[i]['image_name'])
    captions.append(data[i]['caption'])
    labels.append(data[i]['label'])
    temp = cv2.imread(imagePath+data[i]['image_name'])
    if(i%1000 == 0):
        print('{} images loaded'.format(i))
    images.append(temp)

vector_output = outil.predictImageVector(images)



beSaved = []
for i in range(0,len(data)):
    temp = {}
    temp['image_name'] = imageName[i]
    temp['label'] = labels[i]
    temp['caption'] = captions[i]
    temp['class'] = labelToClass[labels[i]]
    temp['vector'] = vector_output[i]
    beSaved.append(temp)
with open(savePreffix+'ImageVector'+suffix,'w') as f:
    json.dump(beSaved,f)

