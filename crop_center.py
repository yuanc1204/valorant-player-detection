import argparse
import cv2
import os


def center_crop(input_dir, output_dir, crop_size=640):
    """Center-crop every image in input_dir to crop_size x crop_size and save to output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f'Found {len(images)} images, center-cropping...')

    saved = 0
    for img_name in images:
        img = cv2.imread(os.path.join(input_dir, img_name))
        if img is None:
            print(f'WARN: failed to read {img_name}, skipped')
            continue

        h, w, _ = img.shape
        if h < crop_size or w < crop_size:
            print(f'WARN: image smaller than {crop_size}px, skipped: {img_name} ({w}x{h})')
            continue

        # Center crop with boundary clamping
        left = max(0, w // 2 - crop_size // 2)
        top = max(0, h // 2 - crop_size // 2)
        right = min(w, left + crop_size)
        bottom = min(h, top + crop_size)

        cv2.imwrite(os.path.join(output_dir, img_name), img[top:bottom, left:right])
        saved += 1

    print(f'Done! Cropped {saved}/{len(images)} images to: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Center-crop screenshots to a square. '
                    'Keeps players undistorted compared to resizing the full frame.')
    parser.add_argument('-i', '--input', required=True, help='folder with screenshots (required)')
    parser.add_argument('-o', '--output', default=r'new_data\images', help='output folder (default: new_data/images)')
    parser.add_argument('-s', '--size', type=int, default=640, help='crop size (default: 640)')
    args = parser.parse_args()
    center_crop(args.input, args.output, args.size)
