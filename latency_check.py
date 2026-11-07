import time
from ultralytics import YOLO
import cv2
 
model = YOLO(r"E:\Projects\Weld_Project\runs\detect\weld_detection_project\weld_yolo_training-4\weights\best.pt")          # or the full runs/... path
 
image = cv2.imread("sample.jpg")
 
start = time.time()
results = model(image)
end = time.time()
 
latency = end - start
print("Inference Time:", latency, "seconds")
