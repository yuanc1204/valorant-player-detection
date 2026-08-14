from ultralytics import YOLO

# Export a trained model to TensorRT engine (FP16, faster inference)
model = YOLO(r'models/best.pt')
model.export(format='engine', device=0, half=True)
