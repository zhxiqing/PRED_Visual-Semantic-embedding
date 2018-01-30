import outil
import json
imagePath = '/media/zxq/zxq/COCO/Valresized/'
preffix = './valCaptions/'
suffix = '.json'
labelToClass = {}
with open('./labelToClass.json','r') as f:
    labelToClass = json.load(f)
classes = ['airplane','apple','banana',
            'backpack','stop sign','bed',
            'cup','bus','horse']
filename = []
for c in classes:
    filename.append(preffix+c+suffix)
images,texts,metadata = outil.readDataFromJson(filename,imagePath,isList=True,loadSize=50)
print(len(texts))
vectors = outil.predictTextImageVectorNN(texts,images)
print(vectors[0])
print(len(vectors[0]))