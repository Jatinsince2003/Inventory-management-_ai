import torch
import numpy as np
import cv2 

model = torch.hub.load("ultralytics/yolov5" , "yolov5s")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    rect , frame = cap.read()
    predictions = model(frame)
    cv2.imshow("frame" ,np.squeeze(predictions.render()))
    if (cv2.waitKey(10) & 0xFF == ord('q')):
        break
    