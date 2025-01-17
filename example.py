import os
from inference import DetectFire

image_path = "test_images/candle.jpg"

score = DetectFire.detect(image_path, graph_preds=True)
print(score)