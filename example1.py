import os
from inference import DetectFire

image_path = "test_images/Kitchen.jpg"

#Instanciate 'DetectFire' Class
detector = DetectFire(device='cpu')
#Use 'detect' method to get predictions and graph
score = detector.detect(image_path=image_path, graph_preds=True, threshold=.9)
#Print Score 
print(f"Score: {100*score:.2f}%")
