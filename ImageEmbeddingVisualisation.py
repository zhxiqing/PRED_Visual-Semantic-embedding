import tensorflow as tf
import numpy as np
import json
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.contrib.tensorboard.plugins import projector
import cv2
preffix = './captionInfo/'
suffix = '.json'
classes = ['airplane','apple','banana',
            'backpack','stop sign','bed',
            'cup','bus','horse']
labelToClass = {}
data = []
imageName = []
images = []
LOG_DIR = './VisualisationEmbedding_2/'
path_for_coco_sprites =  'spriteImage900.png'
path_for_coco_metadata =  'metadata900.tsv'
NAME_TO_VISUALISE_VARIABLE = "cocoembedding"
modelPath = './ImageEmbedding.h5'
imagePath = '/media/zxq/zxq/COCO/resized/'
for m_class in classes:
    with open(preffix+m_class+suffix) as f:
        temp = json.load(f)
    labelToClass[temp[0]['label']] = m_class
    data += temp[0:100]
    f.close()
for i in range(0,len(data)):
    imageName.append(data[i]['image_name'])
for i in range(0,len(data)):
    temp = cv2.imread(imagePath+imageName[i])
    images.append(temp)
print(len(images))
images = np.asarray(images)
print(images.shape)
#Normalisation
images = images.astype('float32')
images /= 255
images = list(images)
model_1 = keras.models.load_model(modelPath)
get_layer_output = K.function([model_1.layers[0].input,K.learning_phase()],[model_1.layers[14].output])
layer_output = []
for im in images:
    temp = get_layer_output([[im],0])[0]
    layer_output.append(temp[0].tolist())
layer_output = np.asarray(layer_output)
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

