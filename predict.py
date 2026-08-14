from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Path to your trained weights
    model_path = r'models/best.pt'

    if not os.path.exists(model_path):
        print(f'Error: model not found at {model_path}. Run train.py first.')
        exit()

    print(f'Loading model: {model_path}')
    model = YOLO(model_path)

    # Folder of images to test
    source_images = r'datasets\images\val'

    print('Running batch inference...')
    results = model.predict(
        source=source_images,
        save=True,          # save annotated images
        conf=0.3,           # confidence threshold
        project='runs',
        name='predict',
        exist_ok=True,
        device=0            # GPU index; use 'cpu' if no CUDA torch
    )

    print('Done! Results saved to: runs/predict')
