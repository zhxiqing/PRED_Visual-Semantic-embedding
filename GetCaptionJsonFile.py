from pycocotools.coco import COCO
import json
dataDir = '.'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
capFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco = COCO(annFile)
coco_caps=COCO(capFile)
catIds = coco.getCatIds()
catInfo = coco.loadCats(catIds)
catNames= [cat['name'] for cat in catInfo]
nameIndex=0
total = []

# For each class, create the caption json file
for cid in catIds:
    filename = catNames[nameIndex]
    nameIndex=nameIndex+1
    imageIds = coco.getImgIds(catIds=cid)
    print('id : {} name : {}'.format(cid,filename))
    capIds = coco_caps.getAnnIds(imgIds=imageIds)
    captions = coco_caps.loadAnns(capIds)
    for i in range(0,len(captions)):
        captions[i]['label'] = cid-1
    with open('./captionInfo/{}.json'.format(filename),'w') as f:
        json.dump(captions,f)

# Create a seul json file for all class
#print(catIds)
#for cid in catIds:
#    imageIds=coco.getImgIds(catIds=cid)
#    total = total + imageIds
#    capIds = coco_caps.getAnnIds(imgIds=imageIds)
#    captions = coco_caps.loadAnns(capIds)
#    for i in range(0,len(captions)):
#        captions[i]['label'] = cid-1
#print('Size of all images {}'.format(len(total)))
#capIds = coco_caps.getAnnIds(imgIds=total)
#print('Size of all captions {}'.format(len(capIds)))
#captions = coco_caps.loadAnns(capIds)
#with open('./captionInfo/AllCaptions_0begin.json','w') as f:
#    json.dump(captions,f)