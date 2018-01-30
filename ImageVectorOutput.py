import json
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
import random
import numpy as np
from keras import backend as K
#from PIL import Image
import matplotlib.pyplot as plt
preffix = './captionInfo/'
suffix = '.json'
classes = ['airplane','apple','banana',
            'backpack','stop sign','bed',
            'cup','bus','horse']
labelToClass = {}
data = []
imageName = []
images = []
captions = []
labels = []
modelPath = './ImageEmbedding.h5'
imagePath = '/media/zxq/zxq/COCO/resized/'
sprite_lenth = 45
for m_class in classes:
    with open(preffix+m_class+suffix) as f:
        temp = json.load(f)
    labelToClass[temp[0]['label']] = m_class
    print(len(temp))
    data += temp[0:100]
    f.close()
for i in range(0,len(data)):
    imageName.append(data[i]['image_name'])
    captions.append(data[i]['caption'])
    labels.append(data[i]['label'])
for i in range(0,len(data)):
    temp = cv2.imread(imagePath+imageName[i])
    images.append(temp)
print(len(images))

images = np.asarray(images)
beSaved = images
print(images.shape)
#Normalisation
images = images.astype('float32')
images /= 255
images = list(images)
model_1 = keras.models.load_model(modelPath)

#get_layer_output = K.function([model_1.layers[0].input,K.learning_phase()],[model_1.layers[14].output])
#layer_output = get_layer_output([images,0])[0]
#print('Layer OK')

#with open('ImageVector2025.tsv','w') as vectorFile:
#    for i in range(0,layer_output.shape[0]):
#        for j in range(0,layer_output.shape[1]-1):
#            vectorFile.write(str(layer_output[i][j])+'\t')
#        vectorFile.write(str(layer_output[i][-1])+'\n')
#    vectorFile.close()
with open('metadata2025.tsv','w') as metaData:
    metaData.write('Label\tCaption\n')
    for i in range(0,len(captions)):
        tmp = captions[i].replace('\n','')
        metaData.write(labelToClass[labels[i]]+'\t'+tmp+'\n')
    metaData.close()
#sprite_width = 32*sprite_lenth
#sprite_high = 32*sprite_lenth
#sprite_image = Image.new('RGBA',(int(sprite_width),int(sprite_high)))
#print("created")
#for count,im in enumerate(beSaved):
#    locationX = (count % sprite_lenth)*32
#    locationY = int(count / sprite_lenth)*32
#    box = (locationX,locationY,locationX+32,locationY+32)
#    print(type(box))
#    im = Image.fromarray(im)
#    im = im.resize((32,32),Image.ANTIALIAS)
#    sprite_image.paste(im,box)
#print("Done")
#print("Saving:")
#sprite_image.save('spriteImage2025.png')
#print("saved!")
def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots, 3),type(images[0][0][0][0]))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img
    
    return spriteimage

sprite_image = create_sprite_image(beSaved)
plt.imsave('spriteImage2025.png',sprite_image)