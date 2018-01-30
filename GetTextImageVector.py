import numpy as np
import json
import pickle
import cv2
import random
import keras
import keras.preprocessing.text
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
import matplotlib.pyplot as plt
from sklearn import manifold

captionPath=['./valCaptions/airplane.json','./valCaptions/apple.json','./valCaptions/banana.json',
            './valCaptions/backpack.json','./valCaptions/stop sign.json','./valCaptions/bed.json',
            './valCaptions/cup.json','./valCaptions/bus.json','./valCaptions/horse.json']
#captionPathTotal = './captionInfo/AllCaptionsOneCaptionperImage.json'
# Load caption file
data = []
Ids = [] 
for Path in captionPath:
    with open(Path,'r') as f:
        temp = json.load(f)
    Ids.append(temp[0]['label'])
    print("{}:{}".format(Path,temp[0]['label']))
    data += temp
    f.close()
j = 0
mapColor = {}
for i in Ids:
    j+=1
    mapColor[i] = j
tokenPath = './tokenizerV2.pickle'
textModelPath = './TextEmbeddingL100.h5'
imageModelPath = './ImageEmbedding.h5'
imagePath = '/media/zxq/zxq/COCO/resized/'
vectorModelPath = './TextImageEmbedding.h5'

text = []
imageName = []
labels = []
instanceSize = len(data)
random.shuffle(data)
#Load captions and images' name
for i in range(0,instanceSize):
    text.append(data[i]['caption'])
    imageName.append(data[i]['image_name'])
    labels.append(data[i]['label'])
#Load tokenizer from file
with open('./tokenizerV2.pickle','rb') as handle:
    Tokenizer=pickle.load(handle)

#Pre-processing of texts
max_caption_length=400
text = Tokenizer.texts_to_sequences(text)
text = np.asarray(text)
text = sequence.pad_sequences(text, maxlen=max_caption_length)

#Load text and image classification model
model_1 = keras.models.load_model(textModelPath) 
model_2 = keras.models.load_model(imageModelPath)
model_3 = keras.models.load_model(vectorModelPath)

#Function to get intermediate layer output of image, text and vector
get_layer_output_image = K.function([model_2.layers[0].input,K.learning_phase()],[model_2.layers[14].output])
get_layer_output_text = K.function([model_1.layers[0].input],[model_1.layers[1].output])
get_layer_output_vector = K.function([model_3.layers[0].input],[model_3.layers[0].output])


#Get the intermediate vector
intertVectors = []
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
    intertVectors.append(vector)
conVectors = np.asarray(intertVectors)
print(conVectors.shape)
outPut = get_layer_output_vector([intertVectors])[0]
print(outPut.shape)
print(type(outPut))


# Scale and visualize the embedding vectors                            
def plot_embedding(X, title=None):     
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)                           

    plt.figure()              
    ax = plt.subplot(111)      
    for i in range(X.shape[0]):                        
        plt.text(X[i, 0], X[i, 1], str(labels[i]),    
             color=plt.cm.Set1(mapColor[labels[i]]/10.0),         
             fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

# t-SNE embedding
print("Computing t-SNE embedding")           
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(outPut)
plot_embedding(X_tsne,                                   
            "t-SNE embedding of the final vectors(a part of labels) ")                           
plt.show() 