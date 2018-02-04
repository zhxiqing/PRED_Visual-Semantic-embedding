import outil
import json
testPath = '/media/zxq/zxq/COCO/Valresized/'
trainPath = '/media/zxq/zxq/COCO/resized/'
testPreffix = '../valCaptions/'
trainPreffix = '../captionInfo/'
suffix = '.json'
labelToClass = {}
trainSavedPath = '../128N200LSTM/Train/'
testSavedPath  = '../128N200LSTM/Test/'
with open('./labelToClass.json','r') as f:
    labelToClass = json.load(f)
classes = ['airplane','apple','banana',
            'backpack','stop sign','bed',
            'cup','bus','horse']
filename = []
for c in classes:
    filename.append(trainPreffix+c+suffix)
trainImages,trainTexts,trainMetadata = outil.readDataFromJson(filename,trainPath,isList=True,loadSize=2000)
print(len(trainTexts))
imageVectors = outil.predictImageVector(trainImages)
TextVectors = outil.predictTextVector(trainTexts)
imageTextVectors = []
for i in range(0,len(TextVectors)):
    imageTextVectors.append(TextVectors[i]+imageVectors[i])
print("Train Vector calculated")
outil.saveToJson(trainSavedPath+'ImageVec.json',trainMetadata,imageVectors,labelToClass)
outil.saveToJson(trainSavedPath+'TextVec.json',trainMetadata,TextVectors,labelToClass)
outil.saveToJson(trainSavedPath+'ImageTextVec.json',trainMetadata,imageTextVectors,labelToClass)

filename = []
for c in classes:
    filename.append(testPreffix+c+suffix)
testImages,testTexts,testMetadata = outil.readDataFromJson(filename,testPath,isList=True,loadSize=5000)
print(len(testTexts))
imageVectors = outil.predictImageVector(testImages)
TextVectors = outil.predictTextVector(testTexts)
imageTextVectors = []
for i in range(0,len(TextVectors)):
    imageTextVectors.append(TextVectors[i]+imageVectors[i])
print("Test Vector calculated")
outil.saveToJson(testSavedPath+'ImageVec.json',testMetadata,imageVectors,labelToClass)
outil.saveToJson(testSavedPath+'TextVec.json',testMetadata,TextVectors,labelToClass)
outil.saveToJson(testSavedPath+'ImageTextVec.json',testMetadata,imageTextVectors,labelToClass)