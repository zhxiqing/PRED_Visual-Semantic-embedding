import json
import keras
import keras.preprocessing.text
import numpy as np
import outil

captionPath = '../valCaptions/AllCaptionsOneCaptionperImage.json'
captions, labels = outil.readEvaluationTextFromJson(captionPath,False,20000)

model0 = keras.models.load_model('../TextEmbeddingL100.h5')
model1 = keras.models.load_model('../TextEmbedding200L.h5')

score0 = model0.evaluate(captions,labels)
score1 = model1.evaluate(captions,labels)

print("100L loss:{}, 100L accuracy:{}".format(score0[0],score0[1]))
print("200L loss:{}, 200L accuracy:{}".format(score1[0],score1[1]))