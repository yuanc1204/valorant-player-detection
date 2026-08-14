from ultralytics import YOLO

if __name__ == '__main__':
    # Load a base model (auto-downloads if missing) or your own trained weights
    model = YOLO('yolo11n.pt')

    # Train / fine-tune
    model.train(
        data='data.yaml',   # dataset config (see data.yaml)
        imgsz=640,          # input size (matches the 640x640 center crops)
        epochs=100,
        batch=40,           # lower this if you run out of GPU memory
        device=0,           # GPU index; use 'cpu' if no CUDA torch
        workers=4,
        project='runs'
    )
