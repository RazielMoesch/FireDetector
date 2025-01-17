from model import FireDetector
from PIL import Image
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from matplotlib import patches



class DetectFire():
  def __init__(self, model_path):
    self.detector = FireDetector(model_path, 2)
  
  def preprocess_image(self, image_path):
    image = Image.open(image_path).convert("RGB")
    image = F.to_tensor(image).unsqueeze(0)
    return image
  
  def detect(self, image_path, graph_preds=False, threshold=.8):
    image = self.preprocess_image(image_path)
    model = self.detector.model
    model.eval()
    with torch.inference_mode():
      preds = model(image)

    if graph_preds:
      self.graph_preds(image, preds, threshold)
    
    score = self.preds_to_results(preds)

    return score

  def graph_preds(self, image, preds, threshold):
    image = image.permute(1,2,0).cpu().numpy()
    boxes = preds[0]["boxes"].cpu().numpy()
    scores = preds[0]["scores"].cpu().numpy()
    fig, ax = plt.subplots()
    ax.imshow(image)

    for box, score in zip(boxes, scores):
      if score > threshold:
        xmin, ymin, xmax, ymax = box
        w = xmax - xmin
        h = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='lightblue', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'Score: {score:.2f}', color='lightblue', fontsize= 12)
    plt.show()
  
  def preds_to_results(self, preds):
    score = torch.max(preds[0]["scores"])
    return score
  
  #change