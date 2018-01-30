import json
import numpy as np

loadPath = './VectorJson/neighbors/ImageKNN.json'

with open(loadPath) as f:
    data = json.load(f)

correctPrediction = 0
perfectTimes = 0
goodTimes = 0
for instance in data:
    tmp = 0
    for nb in instance['neighbors']:
        if(instance['label'] == nb['label']):
            correctPrediction+=1
            tmp += 1     
    if(tmp == 10):
        perfectTimes+=1
    if(tmp >= 5):
        goodTimes+=1

print("For {} instance, all predictions have the same label.".format(perfectTimes))
print("For {} instance, more than a half of predictions have the same label.".format(goodTimes))
print("Total correct predictioin: {} of {}".format(correctPrediction,len(data)*10))
print("Perfect prediction percent: {}%".format(perfectTimes/len(data)*100))
print("Good prediction percent: {}%".format(goodTimes/len(data)*100))
print("Correct prediction percent: {}%".format(correctPrediction/len(data)*10))

#Precision et recall   MAP (mean average precision)
#information retrival scki-learn
#Pour 4 type de vecteurs(Text, Image, Image+Text, Image+Text de sortie de reseaux neural) sur des mesures differents, 
#et voir , comparer les resultats.
#Lundi matin 10h 