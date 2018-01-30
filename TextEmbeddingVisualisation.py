import tensorflow as tf
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
from tensorflow.contrib.tensorboard.plugins import projector
preffix = './captionInfo/'
suffix = '.json'
captionPath=['airplane','apple','banana',
            'backpack','stop sign','bed',
            'cup','bus','horse']
tokenPath = './tokenizerV2.pickle'
modelPath = './TextEmbeddingL100.h5'
LOG_DIR = './VisualisationEmbedding_3/'
path_for_coco_sprites =  'spriteImage900.png'
path_for_coco_metadata =  'metadata900.tsv'
NAME_TO_VISUALISE_VARIABLE = "cocoembeddingText"
labelToClass = {}
data = []
for m_class in captionPath:
    with open(preffix+m_class+suffix) as f:
        temp = json.load(f)
    labelToClass[temp[0]['label']] = m_class
    data+=temp[0:100]
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


#Creating the embeddings
embedding_var = tf.Variable(layer_output,name = NAME_TO_VISUALISE_VARIABLE)
summary_writer = tf.summary.FileWriter(LOG_DIR)

#Create the embedding projector
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

embedding.metadata_path = path_for_coco_metadata
embedding.sprite.image_path = path_for_coco_sprites
embedding.sprite.single_image_dim.extend([128,128])

# Say that you want to visualise the embeddings
projector.visualize_embeddings(summary_writer,config)

#Saving the data
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess,LOG_DIR+'checkPoint.ckpt',1)