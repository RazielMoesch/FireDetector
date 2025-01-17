import os
from inference import DetectFire

image_path = "test_images/candle.jpg"

detector = DetectFire(device='cpu')
score = detector.detect(image_path=image_path, graph_preds=True)
print(f"Score: {float(100*score):.2f}%")