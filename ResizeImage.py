import json
import cv2
preffix = './valCaptions/'
suffix = '.json'
allImages = 'AllCaptionsOneCaptionperImage'
savePreffix = '/media/zxq/zxq/COCO/Valresized/'
imagePath = '/media/zxq/zxq/COCO/val2017/'
with open(preffix+allImages+suffix) as f:
    data = json.load(f)
image_name = [] 
for i in range(0,len(data)):
    temp = cv2.imread(imagePath+data[i]['image_name'])
    resized = cv2.resize(temp,(128,128),interpolation=cv2.INTER_AREA)
    cv2.imwrite(savePreffix+data[i]['image_name'],resized)
    if(i % 100 == 0):
        print("{} images treated!".format(i))
      

