import cv2
from ultralytics import YOLO
img_pth = "train/images/20221224131640_016701_MP4-13_jpg.rf.b68daddfb32e9edd568d74aec3129482.jpg"
model = YOLO("runs/segment/train/weights/best.pt")
results = model(source=img_pth)
res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)

