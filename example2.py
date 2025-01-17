import os
from inference import DetectFire

image_path = "test_images/candle.jpg"

#Instanciate 'DetectFire' Class
detector = DetectFire(device='cpu')
#Use 'detect' method to get predictions (including bounding boxes) and graph
score, boxes = detector.detect(image_path=image_path, return_boxes=True, graph_preds=True, threshold=.9)
#Print results (remember its in the form of tensors)
print(score, boxes)