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
from keras import backend as K
preffix = './valCaptions/'
suffix = '.json'
captionPath=['airplane','apple','banana',
            'backpack','stop sign','bed',
            'cup','bus','horse']

tokenPath = './tokenizerV2.pickle'
modelPath = './TextEmbeddingL100.h5'
labelToClass = {}
data = []
for m_class in captionPath:
    with open(preffix+m_class+suffix) as f:
        temp = json.load(f)
    labelToClass[temp[0]['label']] = m_class
    data+=temp
    f.close()
test_text = []
test_label = []
for i in range(0,len(data)):
    test_text.append(data[i]['caption'])
    test_label.append(data[i]['label'])
captions = test_text
classes = test_label
with open(tokenPath,'rb') as handle:
    Tokenizer=pickle.load(handle)

max_caption_length=400
test_text = Tokenizer.texts_to_sequences(test_text)
test_text = np.asarray(test_text)
test_label = np.asarray(test_label)
test_text = sequence.pad_sequences(test_text, maxlen=max_caption_length)
test_label = to_categorical(test_label,num_classes=90)

model_1 = keras.models.load_model(modelPath)
#scores = model_1.evaluate(test_text, test_label, verbose=0)
#predict_output = model_1.predict(test_text)
#print("Accuracy: %.2f%%" % (scores[1]*100))
#print(predict_output[0])
#print(test_label[0])

#Get intermediate layer output
print(test_text.shape)
get_layer_output = K.function([model_1.layers[0].input],[model_1.layers[1].output])
layer_output = get_layer_output([test_text])[0]
print(layer_output.shape)
#print(layer_output)
scores = model_1.evaluate(test_text, test_label, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

with open('TextVector.tsv','w') as vectorFile:
    for i in range(0,layer_output.shape[0]):
        for j in range(0,layer_output.shape[1]-1):
            vectorFile.write(str(layer_output[i][j])+'\t')
        vectorFile.write(str(layer_output[i][-1])+'\n')
    vectorFile.close()

with open('metadata.tsv','w') as metaData:
    metaData.write('Label\tCaption\n')
    for i in range(0,len(captions)):
        tmp = captions[i].replace('\n','')
        metaData.write(labelToClass[classes[i]]+'\t'+tmp+'\n')
    metaData.close()