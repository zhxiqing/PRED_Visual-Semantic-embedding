import json
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
import random
import numpy as np
dataPath = '/media/zxq/zxq/COCO/resized/'
captionPath = './captionInfo/AllCaptionsOneCaptionperImage.json'
modelPath = './ImageEmbedding2.h5'
trainNumber = 1000
testNumber = 200
numClass = 90
# Load caption file 
with open(captionPath,'r') as f:
    data = json.load(f)
imageName = []
labels = []
images = []
m_labels = []
trainImages = []
trainLabels = []
valImages = []
valLabels = []
#Get the images' name and labels
print('Data lenth {}'.format(len(data)))
for i in range(0,len(data)):
    imageName.append(data[i]['image_name'])
    labels.append(data[i]['label'])

#Shuffle the images and corresponding label
beShuffle = list(zip(imageName,labels))
random.shuffle(beShuffle)
imageName,labels = zip(*beShuffle)
imageName = list(imageName)
labels = list(labels)

#Devide images into train set and validation set.
for i in range(0,trainNumber):
    temp = cv2.imread(dataPath+imageName[i])
    if(temp is not None):
        trainImages.append(temp)
        trainLabels.append(labels[i])
for j in range(trainNumber,trainNumber + testNumber):
    temp = cv2.imread(dataPath+imageName[j])
    if(temp is not None):
        valImages.append(temp)
        valLabels.append(labels[j])

trainImages = np.asarray(trainImages)
trainLabels = np.asarray(trainLabels)
valImages = np.asarray(valImages)
valLabels = np.asarray(valLabels)
print(trainImages.shape)
print(valImages.shape)

#Convert labels to one-hot.
trainLabels = to_categorical(trainLabels)
valLabels = to_categorical(valLabels)
print(trainLabels[0].shape)
print(valImages[0].shape)
#Create model
model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same',
                 input_shape=trainImages.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (5, 5),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(numClass))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#Complie the model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
#Model information
print(model.summary())
#Normalisation
trainImages = trainImages.astype('float32')
valImages = valImages.astype('float32')
trainImages /= 255
valImages /= 255
#Star training
batch_size = 64
epochs = 4
model.fit(trainImages, trainLabels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(valImages, valLabels),
        shuffle=True)

model.save(modelPath)
scores = model.evaluate(valImages, valLabels, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
