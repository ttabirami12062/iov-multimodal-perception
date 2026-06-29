from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import cv2

model = YOLO("yolov8m.pt")

results = model.predict(source="G:/project code january/YOLO_ROAD_SEGMEN/train/images/20221224131640_016701_MP4-8_jpg.rf.51a9690c56ff63a2ba449e787913c59f.jpg", show=True)

print(results)