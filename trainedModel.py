# importing libraries
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


# load model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path="yolov5/runs/train/exp6/weights/best.pt", force_reload=True)

# predictions = model("photo4.jpg")
# predictions.save()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    rect, frame = cap.read()
    results = model(frame)
    cv2.imshow("video Capture", np.squeeze(results.render()))
    if (cv2.waitKey(10) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
