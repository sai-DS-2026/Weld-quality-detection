from ultralytics import YOLO
 
# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")
 
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,                       # use 0 if you have a CUDA GPU
    project="weld_detection_project",
    name="weld_yolo_training"
)

