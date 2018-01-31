import outil
import json
testPath = '/media/zxq/zxq/COCO/Valresized/'
trainPath = '/media/zxq/zxq/COCO/resized/'
testPreffix = '../valCaptions/'
trainPreffix = '../captionInfo/'
suffix = '.json'
labelToClass = {}
trainSavedPath = '../Vector/Train/'
testSavedPath  = '../Vector/Test/'
with open('./labelToClass.json','r') as f:
    labelToClass = json.load(f)
classes = ['airplane','apple','banana',
            'backpack','stop sign','bed',
            'cup','bus','horse']
filename = []
#for c in classes:
#    filename.append(trainPreffix+c+suffix)
#trainImages,trainTexts,trainMetadata = outil.readDataFromJson(filename,trainPath,isList=True,loadSize=2000)
#print(len(trainTexts))
#trainVectors = outil.predictTextImageVector(trainTexts,trainImages)
#print("Vector calculated")
#outil.saveToJson(trainSavedPath+'TextImage.json',trainMetadata,trainVectors,labelToClass)


for c in classes:
    filename.append(testPreffix+c+suffix)
testImages,testTexts,testMetadata = outil.readDataFromJson(filename,testPath,isList=True,loadSize=5000)
print(len(testTexts))
testVectors = outil.predictTextImageVector(testTexts,testImages)
print("Vector calculated")
outil.saveToJson(testSavedPath+'TextImage.json',testMetadata,testVectors,labelToClass)