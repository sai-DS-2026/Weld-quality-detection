# from ultralytics import YOLO
 
# model = YOLO("runs/detect/weld_detection_project/weld_yolo_training-4/weights/best.pt")
 
# metrics = model.val()
# print(metrics)

from ultralytics import YOLO

model = YOLO(
    r"runs/detect/weld_detection_project/weld_yolo_training-4/weights/best.pt"
)

metrics = model.val(    
    data="data.yaml",
    imgsz=640,
    batch=1,
    device=0,
    workers=0)

print(metrics)
