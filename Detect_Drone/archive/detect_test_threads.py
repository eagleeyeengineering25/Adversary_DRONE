"""
Threaded YOLOv5 detection with webcam

Usage:
    $ python detect_threaded.py --weights yolov5s.pt --source 0 --imgsz 320 --view-img
"""

import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import argparse
import sys
import time
from pathlib import Path
from queue import Queue
import threading

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.general import (check_img_size, check_requirements, check_imshow,
                           colorstr, non_max_suppression, scale_coords,
                           xyxy2xywh, increment_path, set_logging)
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized

# -----------------------------
# Threaded camera class
class CameraThread(threading.Thread):
    def __init__(self, src=0, queue_size=5):
        super().__init__()
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.queue = Queue(maxsize=queue_size)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if not self.queue.full():
                self.queue.put(frame)

    def read(self):
        if self.queue.empty():
            return None
        return self.queue.get()

    def stop(self):
        self.running = False
        self.cap.release()
# -----------------------------

@torch.no_grad()
def run(weights='yolov5s.pt', source='0', imgsz=640, conf_thres=0.25,
        iou_thres=0.45, max_det=1000, device='', view_img=True,
        save_txt=False, save_conf=False, save_crop=False,
        nosave=False, classes=None, agnostic_nms=False,
        augment=False, update=False, project='runs/detect',
        name='exp', exist_ok=False, line_thickness=3,
        hide_labels=False, hide_conf=False, half=False):

    webcam = source == '0'  # only support threaded webcam for '0'

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()

    # Warmup
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    # Start threaded camera
    cam_thread = None
    if webcam:
        cam_thread = CameraThread(0)
        cam_thread.start()

    fps_counter = 0
    t0 = time.time()

    try:
        while True:
            if webcam:
                im0 = cam_thread.read()
                if im0 is None:
                    continue
            else:
                raise NotImplementedError("Only threaded webcam implemented in this version.")

            h, w = im0.shape[:2]

            # Draw center dot
            cv2.circle(im0, (w//2, h//2), 5, (0, 0, 255), -1)

            # Preprocess for YOLO
            
            # img = cv2.resize(im0, (imgsz, imgsz))
            # img = img[:, :, ::-1].copy()  # BGR → RGB
            # img = torch.from_numpy(img).to(device)
            # img = img.half() if half else img.float()
            # img /= 255.0
            # if img.ndimension() == 3:
                # img = img.unsqueeze(0)
                
            # Convert to 3 channels if needed
            if im0.ndim == 2:  # grayscale
                im0 = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)

            # Resize and convert BGR to RGB
            img = cv2.resize(im0, (imgsz, imgsz))
            img = img[:, :, ::-1].copy()  # BGR → RGB
            img = torch.from_numpy(img).to(device)
            img = img.permute(2, 0, 1).contiguous()  # HWC → CHW
            img = img.half() if half else img.float()
            img /= 255.0  # 0-255 → 0.0-1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)  # add batch dimension


            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_synchronized()

            # Process detections
            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = map(int, xyxy)
                        x_center = int((x1 + x2)/2)
                        y_center = int((y1 + y2)/2)

                        # Draw center of detection
                        cv2.circle(im0, (x_center, y_center), 5, (0, 255, 0), -1)
                        # Draw bounding box
                        plot_one_box(xyxy, im0, label=f'{names[int(cls)]} {conf:.2f}',
                                     color=colors(int(cls), True), line_thickness=line_thickness)
                        # Print coordinates
                        print(f"Detected {names[int(cls)]} at (x={x_center}, y={y_center}) conf={conf:.2f}")

            # Display FPS
            fps_counter += 1
            if fps_counter % 10 == 0:
                fps = 10 / (time.time() - t0)
                t0 = time.time()
                cv2.putText(im0, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if view_img:
                cv2.imshow("YOLOv5 Threaded", im0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        if cam_thread:
            cam_thread.stop()
        cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt

def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
