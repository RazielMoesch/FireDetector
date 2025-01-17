import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor




class FireDetector():
  def __init__(self, saved_dict_path=os.path.join('models', 'firedetectordict1.pth'), num_classes=2):
    self.saved_dict = saved_dict_path
    self.num_classes = num_classes
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.load_model()


  def load_model(self):
    self.model = self.get_model(self.num_classes)
    self.model.to(self.device)
    if os.path.exists(self.saved_dict):
      self.model.load_state_dict(torch.load(self.saved_dict, map_location=self.device)) 
      
    else:
      raise FileNotFoundError(f"{self.saved_dict} was not found.")

  def get_model(self, num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
  