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
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
path = './captionInfo/AllCaptionsOneCaptionperImage.json'
runName = 'TextEmbeddingL100_2/'
logdir = './logs/{}'.format(runName)
TextLabelPath = 'metadata.tsv'
# Load caption file 
with open(path,'r') as f:
    data = json.load(f)

#Create list for train/test texts and labels
text=[]
label=[]
train_text = []
train_label = []
test_text=[]
test_label=[]

#Get the data
print('Data lenth {}'.format(len(data)))
for i in range(0,len(data)):
    text.append(data[i]['caption'])
    label.append(data[i]['label'])

#Shuffle the data(Each phrase and its label)
beShuffle = list(zip(text,label))
random.shuffle(beShuffle)
text,label = zip(*beShuffle)
text = list(text)
label = list(label)

#Tokenizer and sequencier the captions
#Tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000)
#Fit tokenizer by using 'text'
#Tokenizer.fit_on_texts(text)

#Save tokenizer for test process
#with open('./tokenizerV2.pickle','wb') as handle:
#    pickle.dump(Tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)

#Load tokenizer from file
with open('./tokenizerV2.pickle','rb') as handle:
    Tokenizer=pickle.load(handle)

#Split data to two part, train and test 
spliteIndex = int(len(text)*0.1)
train_text = text[:spliteIndex]
train_label = label[:spliteIndex]
test_text = text[spliteIndex:spliteIndex+10000]
test_label = label[spliteIndex:spliteIndex+10000]
#Save the metadata file
with open(logdir+TextLabelPath,'w') as f:
    f.write("Description\tLabel\n")
    for i in range(0,len(train_label)):
        f.write(train_text[i]+'\t'+str(train_label[i])+'\n')
print(train_text[0])
#Transfer text to index sequences
train_text = Tokenizer.texts_to_sequences(train_text)
test_text = Tokenizer.texts_to_sequences(test_text)

train_text = np.asarray(train_text)
train_label = np.asarray(train_label)
test_text = np.asarray(test_text)
test_label = np.asarray(test_label)

#Convert integer labels to binary labels(One-hot)
print(train_label[0])
train_label = to_categorical(train_label)
test_label = to_categorical(test_label) 
print(train_label.shape)
print(test_label.shape)
print(train_label[0])

# truncate and pad input sequences
max_caption_length = 400
train_text = sequence.pad_sequences(train_text, maxlen=max_caption_length)
test_text = sequence.pad_sequences(test_text, maxlen=max_caption_length)
print(train_text.shape)
print(test_text.shape)
print(train_text[0])

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(5001, embedding_vecor_length, input_length=max_caption_length, mask_zero=True,name = 'embedding'))
model.add(LSTM(100,name = 'LSTM'))
model.add(Dense(90, activation='softmax',name = 'Dense'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
#Tensorboard
embeddingMetadata = {'LSTM':'metadata.tsv'}
tensorBoard = TensorBoard(log_dir=logdir,
                            histogram_freq=1,
                            write_graph=True,
                            write_images=False,
                            embeddings_freq=1,
                            embeddings_layer_names=['embedding','LSTM','Dense'],
                            embeddings_metadata=embeddingMetadata)


#Train the model 
model.fit(train_text, train_label, validation_data=(test_text, test_label), epochs=3, batch_size=64,callbacks=[tensorBoard])
# Final evaluation of the model
scores = model.evaluate(test_text, test_label, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print('Save model:...')
model.save('./logs/100L.h5')
