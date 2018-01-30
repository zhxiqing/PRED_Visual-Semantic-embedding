import json
import keras
import keras.preprocessing.text
import numpy as np
import random
import pickle
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2

captionPath = './captionInfo/AllCaptionsOneCaptionperImage.json'
tokenPath = './tokenizerV2.pickle'
textModelPath = './TextEmbeddingL100.h5'
imageModelPath = './ImageEmbedding.h5'
imagePath = '/media/zxq/zxq/COCO/resized/'
savePath = './TextImageEmbedding.h5'
# Load caption file 
with open(captionPath,'r') as f:
    data = json.load(f)
text = []
imageName = []
labels = []
trainVector = []
instanceSize = 340000
print('Train model in {} instances'.format(instanceSize))
validationSize = 10000

#Load captions and images' name
for i in range(0,instanceSize):
    text.append(data[i]['caption'])
    imageName.append(data[i]['image_name'])
    labels.append(data[i]['label'])
#Load tokenizer from file
with open('./tokenizerV2.pickle','rb') as handle:
    Tokenizer=pickle.load(handle)
#Pre-processing of labels and texts
max_caption_length=400
text = Tokenizer.texts_to_sequences(text)
text = np.asarray(text)
labels = np.asarray(labels)
text = sequence.pad_sequences(text, maxlen=max_caption_length)
labels = to_categorical(labels,num_classes=90)
#Get the train label and validation label
trainLabels = labels[:instanceSize-validationSize]
valLabels = labels[instanceSize-validationSize:]
#Load text and image classification model
model_1 = keras.models.load_model(textModelPath) 
model_2 = keras.models.load_model(imageModelPath)
#Function to get intermediate layer output of image and text
get_layer_output_image = K.function([model_2.layers[0].input,K.learning_phase()],[model_2.layers[14].output])
get_layer_output_text = K.function([model_1.layers[0].input],[model_1.layers[1].output])

for i in range(0,instanceSize):
    if(i%1000 == 0):
        print('{} instance traited'.format(i))
    textTmp = get_layer_output_text([[text[i]]])[0]
    textTmp = textTmp.round(6)
    textTmp = textTmp.tolist()
    imageTmp = cv2.imread(imagePath+imageName[i])
    imageTmp = imageTmp.astype('float32')
    imageTmp /= 255
    imageTmp = get_layer_output_image([[imageTmp],0])[0]
    imageTmp = imageTmp.tolist()
    vector = textTmp[0]+imageTmp[0]
    trainVector.append(vector)
print(len(trainVector))
#Devide dataset to validation set and train set
valVector = np.asarray(trainVector[instanceSize-validationSize:])
trainVector = np.asarray(trainVector[:instanceSize-validationSize])

model = Sequential()
model.add(Dense(100,activation='relu',input_shape = (228,)))
model.add(Dense(90,activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(trainVector, trainLabels,
                    batch_size=64,
                    epochs=3,
                    verbose=1,
                    validation_data=(valVector, valLabels),
                    shuffle=True)
model.save(savePath)
scores = model.evaluate(valVector, valLabels, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])