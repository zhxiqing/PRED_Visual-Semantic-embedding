import json
import sklearn.metrics as skmt
import numpy as np
pureText = '../Neighbors/TextKNN.json'
pureImage = '../Neighbors/ImageKNN.json'
imageText = '../Neighbors/imageTextKNN.json'
imageTextNN = '../Neighbors/NNimageTextKNN.json'

with open(pureText,'r') as f:
    vecText = json.load(f)
    f.close()
with open(pureImage,'r') as f:
    vecImage = json.load(f)
    f.close()
with open(imageText,'r') as f:
    vecImageText = json.load(f)
    f.close()
with open(imageTextNN,'r') as f:
    vecImageTextNN = json.load(f)
    f.close()

def getPrecision(Neighbors,label):
    y_pred = []
    y_true = []
    for n in Neighbors:
        y_pred.append(n['label'])
        y_true.append(label)
    precision = skmt.precision_score(y_true,y_pred,average='micro')
    return precision

def getRecall(Neighbors,label):
    y_pred = []
    y_true = []
    for n in Neighbors:
        y_pred.append(n['label'])
        y_true.append(label)
    recall = skmt.recall_score(y_true,y_pred,average='micro')
    return recall


def getAPS(Neighbors,label):
    precisions = []
    recalls = []
    precisions.append(getPrecision([Neighbors[0]],label))
    recalls.append(getRecall([Neighbors[0]],label))
    for i in range(1,10):
        temp = Neighbors[:i]
        precisions.append(getPrecision(temp,label))
        recalls.append(getRecall(temp,label))
    aps = precisions[0]*recalls[0]
    for i in range(1,10):
        aps+=precisions[i]*(recalls[i]-recalls[i-1])
    return aps

def calPrecision(data):
    num = len(data)
    sumPrecision = 0
    for d in data:
        sumPrecision+=getPrecision(d['neighbors'],d['label'])
    return (sumPrecision/num)

def calmAP(data):
    num = len(data)
    summAP = 0
    for d in data:
        summAP += getAPS(d['neighbors'],d['label'])
    return (summAP/num)


print("Text vector precision : {}".format(calPrecision(vecText)))
print("Image vector precision : {}".format(calPrecision(vecImage)))
print("Image-Text vector precision : {}".format(calPrecision(vecImageText)))
print("Image-Text-NN vector precision : {}".format(calPrecision(vecImageTextNN)))
print("Text vector mAP : {}".format(calmAP(vecText)))
print("Image vector mAP : {}".format(calmAP(vecImage)))
print("Image-Text vector mAP : {}".format(calmAP(vecImageText)))
print("Image-Text-NN vector mAP : {}".format(calmAP(vecImageTextNN)))

#getAPS(vecImage[4]['neighbors'],vecText[0]['label'])

#Reunion Mardi 14h
#变化超参，获得不同的结果