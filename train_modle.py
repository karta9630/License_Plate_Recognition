from ultralytics import YOLO
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    model = YOLO('yolov8n.pt').to(device)
    model.train(
        data='datasets.yaml',  # Path to the dataset configuration file
        epochs=200,            # Number of training epochs
        batch=32,              # Batch size
        workers=2,
        imgsz=(640,384)             # Image size (width and height) for training
        cache=True             # Cache images for faster training
    )
    model = YOLO("runs/detect/train/weights/best.pt").to(device)  # load a pretrained YOLOv8n model
    result = model(source="test/images", save=True)
