import json
import numpy as np
import operator
from scipy import spatial

testPath = '../128N200LSTM/Test/'
trainPath = '../128N200LSTM/Train/'
savePath = '../128N200LSTM/Neighbors/'
suffix = '.json'
inputTrain = 'ImageVec'
inputTest = 'ImageVec'
output = 'ImageVec'


def readJsonFile(fileName):
    with open(fileName) as f:
        data = json.load(f)
    return data

def distanceCosine(a,b):
    return spatial.distance.cosine(a,b)

def distanceEuclidienne(a,b):
    return spatial.distance.euclidean(a,b)

def updateNeighbors(nb,newone):
    sorted_nb = sorted(nb,key=operator.itemgetter(1),reverse = True)
    if sorted_nb[0][1] > newone[1]:
        sorted_nb[0] = newone
    for i in range(0,len(sorted_nb)):
        nb[i] = sorted_nb[i]

def setNeighborsInfo(nb,trainSet):
    neighbors = []
    for i in range(0,len(nb)):
        temp = {}
        temp['image_name'] = trainSet[nb[i][0]]['image_name']
        temp['label'] = trainSet[nb[i][0]]['label']
        temp['caption'] = trainSet[nb[i][0]]['caption']
        temp['distance'] = nb[i][1]
        neighbors.append(temp)
    return neighbors

trainSet = readJsonFile(trainPath+inputTrain+suffix)
testSet = readJsonFile(testPath+inputTest+suffix)

beSaved = []

#For each instance of testSet
for i in range(0,len(testSet)):
    temp = {}
    neighbors = []
    #initialise the neighbors
    for k in range(0,10):
        neighbors.append((-1,3))
    temp['image_name'] = testSet[i]['image_name']
    temp['label'] = testSet[i]['label']
    temp['caption'] = testSet[i]['caption']
    vectorTest = np.array(testSet[i]['vector'])
    #For each instance of trainSet
    for j in range(0,len(trainSet)):
        vectorTrain = np.array(trainSet[j]['vector'])
        disCos = distanceCosine(vectorTest,vectorTrain)
        updateNeighbors(neighbors,(j,disCos))
    temp['neighbors'] = setNeighborsInfo(neighbors,trainSet)
    beSaved.append(temp)
    print("{} of {}".format(i,len(testSet)))

with open(savePath+output+suffix,'w') as f:
    json.dump(beSaved,f)