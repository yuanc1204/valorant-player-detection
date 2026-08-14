"""Extract frames from a match recording with interactive review.

Usage:
    python val_video.py -v <video.mp4> [-m model.engine] [-o output_dir]
"""
import argparse
import cv2
import os
import time
from ultralytics import YOLO


def main(video_path, model_path, save_dir, crop_size, conf_val, jump_seconds, DEFAULT_SKIP):
    if not os.path.exists(video_path):
        print('ERROR: video not found')
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('Loading model...')
    try:
        model = YOLO(model_path)
    except Exception:
        print('ERROR: failed to load model')
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('ERROR: cannot open video')
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    jump_frames = int(jump_seconds * fps)

    print(f'\nVideo FPS: {fps}')
    print('=' * 45)
    print('Controls:')
    print('   [SPACE]     pause/resume (fine-tune with E/Q while paused)')
    print('   [TAB]       toggle fast-skip / frame-by-frame mode')
    print('   [E] / [Q]   next / previous frame (while paused)')
    print('   [D] / [A]   skip forward / backward 5s')
    print('   [S]         save current (cropped) frame')
    print('   [ESC]       quit')
    print('=' * 45)

    paused = False
    frame_skip_rate = DEFAULT_SKIP
    frame_count = 0

    ret, frame = cap.read()
    if not ret:
        return
    last_display_frame = frame

    while True:
        if not paused:
            for _ in range(frame_skip_rate):
                ret, frame = cap.read()
                frame_count += 1
                if not ret:
                    break
            if not ret:
                print('End of video, restarting...')
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        if width >= crop_size and height >= crop_size:
            x1 = (width - crop_size) // 2
            y1 = (height - crop_size) // 2
            crop_img = frame[y1:y1 + crop_size, x1:x1 + crop_size]

            if not paused:
                results = model.predict(crop_img, imgsz=crop_size, conf=conf_val, verbose=False)
                annotated_frame = results[0].plot()
                last_display_frame = annotated_frame
            else:
                annotated_frame = last_display_frame
                if 'need_refresh' in locals() and need_refresh:
                    results = model.predict(crop_img, imgsz=crop_size, conf=conf_val, verbose=False)
                    annotated_frame = results[0].plot()
                    last_display_frame = annotated_frame
                    need_refresh = False

            mode_str = f'SPEED x{frame_skip_rate}' if frame_skip_rate > 1 else 'DETAILED (Slow)'
            status_color = (0, 255, 0) if not paused else (0, 0, 255)
            cv2.putText(annotated_frame, f"{'PAUSED' if paused else 'PLAYING'} | {mode_str}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.imshow('Frame Review', annotated_frame)

        key = cv2.waitKey(1 if not paused else 0)
        key_char = key & 0xFF

        if key == 27:
            break
        elif key_char == ord(' '):
            paused = not paused
            need_refresh = True
        elif key == 9:
            frame_skip_rate = 1 if frame_skip_rate > 1 else DEFAULT_SKIP
            print('Frame-skip mode changed to', frame_skip_rate)
        elif key_char == ord('s'):
            timestamp = int(time.time() * 1000)
            img_name = f'captured_{timestamp}.jpg'
            cv2.imwrite(os.path.join(save_dir, img_name), crop_img)
            print(f'Saved: {img_name}')
        elif key_char == ord('e'):
            if not paused:
                paused = True
            ret, frame = cap.read()
            if ret:
                need_refresh = True
        elif key_char == ord('q'):
            if not paused:
                paused = True
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(pos - 2, 0))
            ret, frame = cap.read()
            if ret:
                need_refresh = True
        elif key_char == ord('d'):
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(pos + jump_frames, total_frames - 1))
            ret, frame = cap.read()
            if ret:
                need_refresh = True
        elif key_char == ord('a'):
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(pos - jump_frames, 0))
            ret, frame = cap.read()
            if ret:
                need_refresh = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract + center-crop frames from a match recording')
    parser.add_argument('-v', '--video', required=True, help='path to the video file (required)')
    parser.add_argument('-m', '--model', default=r'runs/train/weights/best.engine',
                        help='model path (default: runs/train/weights/best.engine)')
    parser.add_argument('-o', '--save-dir', default=r'new_data\images', help='output folder (default: new_data/images)')
    parser.add_argument('--crop-size', type=int, default=640, help='center-crop size (default: 640)')
    parser.add_argument('--conf', type=float, default=0.6, help='detection confidence threshold (default: 0.6)')
    parser.add_argument('--jump-seconds', type=int, default=5, help='seconds to skip with D/A (default: 5)')
    parser.add_argument('--skip', type=int, default=2, help='frame-skip factor (default: 2)')
    args = parser.parse_args()
    main(args.video, args.model, args.save_dir, args.crop_size, args.conf, args.jump_seconds, args.skip)
