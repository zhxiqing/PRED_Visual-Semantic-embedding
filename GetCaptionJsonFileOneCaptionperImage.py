from pycocotools.coco import COCO
import json
dataDir = '.'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
capFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco = COCO(annFile)
coco_caps=COCO(capFile)
catIds = coco.getCatIds()
catInfo = coco.loadCats(catIds)
catNames= [cat['name'] for cat in catInfo]
print(len(catNames))
print(len(catIds))
nameIndex=0
total = []
m_captions = []
# Create a seul json file for all class
print(catIds)
for cid in catIds:
    imageIds = coco.getImgIds(catIds=cid)
    for imID in imageIds:
        str_cap = ''
        m_caption = {}
        capIds = coco_caps.getAnnIds(imgIds=imID)
        captions = coco_caps.loadAnns(capIds)
        for cap in captions:
            str_cap += cap['caption']
        m_caption['caption'] = str_cap
        m_caption['image_id'] = imID
        m_caption['image_name'] = coco.loadImgs(imID)[0]['file_name']
        m_caption['label'] = cid -1
        m_captions.append(m_caption)
print('Size of all captions {}'.format(len(m_captions)))
with open('./valCaptions/AllCaptionsOneCaptionperImage.json','w') as f:
    json.dump(m_captions,f)


#labelMap = dict(zip(catIds,catNames))
#print(labelMap)
#Create one json file for each class
#for cid in catIds:    
#    m_captions = []
#   imageIds = coco.getImgIds(catIds=cid)
#    for imID in imageIds:
#        str_cap = ''
#        m_caption = {}
#        capIds = coco_caps.getAnnIds(imgIds=imID)
#        captions = coco_caps.loadAnns(capIds)
#        for cap in captions:
#        m_caption['caption'] = str_cap
#        m_caption['image_id'] = imID
#        m_caption['image_name'] = coco.loadImgs(imID)[0]['file_name']
#        m_caption['label'] = cid -1
#        m_captions.append(m_caption)
#    saveFile = './captionInfo/{}.json'.format(labelMap[cid])
#    with open(saveFile,'w') as f:
#        json.dump(m_captions,f)
#    f.close()

