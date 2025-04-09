import cv2 
import cvzone
import math
from ultralytics import YOLO


cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)
# find the best.pt file under runs/detect/train*, with * being the latest training run.
model = YOLO("your best.pt model in folder runs* ")

className = ['duy']

while True:
    success, img = cap.read()
    results = model(img, stream= True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Making boxs:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)         
            rong, cao = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, rong, cao))
            # Confident:
            conf = math.ceil((box.conf[0] * 100)) / 100
            # classNAme
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{className[cls]} {conf}', (max(0, x1), max(35, y1)))   


    cv2.imshow("Image", img)
    cv2.waitKey(1)