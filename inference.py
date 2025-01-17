from model import FireDetector
from PIL import Image
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from matplotlib import patches



class DetectFire():

  '''
    A class to detect fire in an image.

    Attributes:
        model_path (str): Path to saved_dict of model. Default: 'models/firedetectordict1.pth'
        device (bool): Sets expected device. Default: Set to whatever is available.
    '''

  def __init__(self, model_path='models/firedetectordict1.pth', device=None):
    
    if not device:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
      self.device = device

    self.detector = FireDetector(model_path, 2, device=self.device)
    
  
  def preprocess_image(self, image_path):
    image = Image.open(image_path).convert("RGB")
    image = F.to_tensor(image).unsqueeze(0)
    return image
  
  def detect(self, image_path, return_boxes=False, graph_preds=False, threshold=.8):
    '''
    Method to perform detection of fire in an image

    Parameters:
        image_path (str): Path to the image for detection
        return_boxes (bool): Decide whether you want the bounding box info to be returned. Default:False
        graph_preds (bool): Decide whether you want a visual represenation of prediction. Default:False
        threshold (float): Number 0-1, only predictions with a higher likelyhood of being fire than the threshold will be graphed and returned
    
    Returns:
        return_boxes:False
          returns the highest score
        return_boxes:True
          returns a tuple of tensors: (scores, boxes)
    '''
    image = self.preprocess_image(image_path).to(self.device)
    model = self.detector.model
    model.eval()
    with torch.inference_mode():
      preds = model(image)

    if graph_preds:
      self.graph_preds(image, preds, threshold)
    
    score = self.preds_to_results(preds)
    if return_boxes:
      filter = preds[0]["scores"] > threshold
      return ((preds[0]["scores"][filter].to(self.device)), (preds[0]["boxes"][filter].to(self.device)))
    else:
      return score.item()

  def graph_preds(self, image, preds, threshold):
    image  = image.squeeze()
    image = image.permute(1,2,0).cpu().numpy()
    boxes = preds[0]["boxes"].cpu().numpy()
    scores = preds[0]["scores"].cpu().numpy()
    fig, ax = plt.subplots()
    ax.set_axis_off()
  
    ax.imshow(image)

    for box, score in zip(boxes, scores):
      if score > threshold:
        xmin, ymin, xmax, ymax = box
        w = xmax - xmin
        h = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='lightblue', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'Score: {(100*score):.2f}%', color='lightblue', fontsize= 12)
    plt.show()
  
  def preds_to_results(self, preds):
    score = torch.max(preds[0]["scores"])
    return score
  
  