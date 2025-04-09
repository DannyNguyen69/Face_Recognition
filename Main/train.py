from ultralytics import YOLO

# Load model YOLOv8n 
model = YOLO("yolov8n.pt")

# Train model
model.train(
    data="/home/dannynguyen/Documents/Huy/yolo_face_dataset/data.yml",  # Đường dẫn dataset.yaml
    epochs=50,           
    imgsz=640,          
    batch=8,              
    device="cpu"         
)