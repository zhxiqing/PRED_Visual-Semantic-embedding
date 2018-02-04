import cv2
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import outil

model0Path = '../ImageEmbedding.h5'
model1Path = '../ImageEmbedding3.h5'
model2Path = '../ImageEmbedding4.h5'
model3Path = '../ImageEmbedding5.h5'
model4Path = '../ImageEmbedding6.h5'
model5Path = '../ImageEmbedding7.h5'
model6Path = '../ImageEmbedding8.h5'
imagePath  = '/media/zxq/zxq/COCO/Valresized/'
captionPath = '../valCaptions/AllCaptionsOneCaptionperImage.json'

images,labels = outil.readEvaluationImageFromJson(captionPath,imagePath,False,20000)
images = np.asarray(images)
labels = np.asarray(labels)
images = images.astype('float32')
images /= 255
labels = to_categorical(labels)

#model0 = keras.models.load_model(model0Path)
model1 = keras.models.load_model(model1Path)
model2 = keras.models.load_model(model2Path)
model3 = keras.models.load_model(model3Path)
model4 = keras.models.load_model(model4Path)
model5 = keras.models.load_model(model5Path)
model6 = keras.models.load_model(model6Path)

#scores0 = model0.evaluate(images, labels,batch_size=100, verbose=1)
scores1 = model1.evaluate(images, labels, verbose=1)
scores2 = model2.evaluate(images, labels, verbose=1)
scores3 = model3.evaluate(images, labels, verbose=1)
scores4 = model4.evaluate(images, labels, verbose=1)
scores5 = model5.evaluate(images, labels, verbose=1)
scores6 = model6.evaluate(images, labels, verbose=1)

#print("Model 4 conv, 128, loss : {}, accuracy : {} ".format(scores0[0],scores0[1]))
print("Model 4 conv, 64, loss : {}, accuracy : {} ".format(scores6[0],scores6[1]))
print("Model 4 conv, 128, loss : {}, accuracy : {} ".format(scores1[0],scores1[1]))
print("Model 4 conv, 256, loss : {}, accuracy : {} ".format(scores2[0],scores2[1]))
print("Model 3 conv, 64, loss : {}, accuracy : {} ".format(scores4[0],scores4[1]))
print("Model 3 conv, 128, loss : {}, accuracy : {} ".format(scores3[0],scores3[1]))
print("Model 3 conv, 256, loss : {}, accuracy : {} ".format(scores5[0],scores5[1]))


