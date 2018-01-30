import json
import keras
import keras.preprocessing.text
import numpy as np
import random
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras import backend as K

preffix = './captionInfo/'
suffix = '.json'
m_class=['airplane','apple','banana',
            'backpack','stop sign','bed',
            'cup','bus','horse']
m_classFull = ['airplane','apple','backpack','banana','baseball bat','baseball glove','bear','bed','bench','bicycle',
                'bird','boat','book','bottle','bowl','broccoli','stop sign','cup','bus','horse']
tokenPath = './tokenizerV2.pickle'
modelPath = './TextEmbeddingL100.h5'
savePreffix = './VectorJson/Train/'
labelToClass = {}
test_text = []
test_label = []
data = []
image_name = []
for className in m_classFull:
    with open(preffix+className+suffix) as f:
        temp = json.load(f)
    labelToClass[temp[0]['label']] = className
    data+=temp[0:2000]
    f.close()


for i in range(0,len(data)):
    test_text.append(data[i]['caption'])
    test_label.append(data[i]['label'])
    image_name.append(data[i]['image_name'])
captions = test_text
with open(tokenPath,'rb') as handle:
    Tokenizer=pickle.load(handle)

max_caption_length=400
test_text = Tokenizer.texts_to_sequences(test_text)
test_text = np.asarray(test_text)
test_text = sequence.pad_sequences(test_text, maxlen=max_caption_length)


model_1 = keras.models.load_model(modelPath)

#Get intermediate layer output
print(len(captions))
print(test_text.shape)
get_layer_output = K.function([model_1.layers[0].input],[model_1.layers[1].output])
layer_output = get_layer_output([test_text])[0]
print(layer_output.shape)

beSaved = []
for i in range(0,len(captions)):
    temp = {}
    temp['image_name'] = image_name[i]
    temp['label'] = test_label[i]
    temp['caption'] = captions[i]
    temp['class'] = labelToClass[test_label[i]]
    temp['vector'] = layer_output[i].tolist()
    beSaved.append(temp)
with open(savePreffix+'TrainVector20'+suffix,'w') as f:
    json.dump(beSaved,f)