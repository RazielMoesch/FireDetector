
import cv2
from inference import DetectFire

### Note: Real time inference is relatively slow on cpu do to its complex framework. ###

#set up camera
cap = cv2.VideoCapture(0)
#instanciate model
detector = DetectFire(device='cpu')

#loop for live feed processing from camera
while True:
  #gets info from camera
  ret, frame = cap.read()
  #just says if image not read stop
  if not ret:
    break
  
  #values from 'detect'
  scores, boxes = detector.detect(frame, return_boxes=True, return_as_np=True, threshold=.8)
  
  #if fire found
  if len(scores) > 0:
    for score,box in zip(scores, boxes):
      print(f"{score*100:.2f}")
      
      xmin, ymin, xmax, ymax = box
      #convert to ints for cv2
      xmin = int(xmin)
      ymin = int(ymin)
      xmax = int(xmax)
      ymax = int(ymax)
      
      #puts info on the feed 
      cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0,0,255), thickness=2)
      cv2.putText(frame, f"{100*score:.2f}%", (xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.5, color=(255,0,0), thickness=2)
  #updates feed to show info
  cv2.imshow("Webcam", frame)
  #if you press 'q' itll stop (must have the window selected)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

#close windows and stop looking at webcam
cap.release()
cv2.destroyAllWindows()