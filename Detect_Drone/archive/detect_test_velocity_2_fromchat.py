#SAVYA

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
    $ python detect_threaded_velocity.py --weights best.pt --source 0 --max-det 1 --imgsz 320

THIS SCRIPT tests sending velocity commands in a threaded architecture

1. YOLO detects target drone -> gets bounding box center (x_center, y_center)
2. Computes error (err_x, err_y) relative to frame center
3. Maps error -> velocity + yaw rate
4. Sends MAVLink messages via Pi to Cube Orange (in a separate thread)
5. Cube Orange adjusts attitude/yaw -> keeps target drone centered
6. Mission Planner shows live response
"""

# added to bypass UnpicklingError: Weights only load failed
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import argparse
import sys
import time
import threading
import queue
from pathlib import Path
from contextlib import suppress

import cv2
import torch
import torch.backends.cudnn as cudnn

# ADDED IMPORTS
from dronekit import connect, VehicleMode, APIException
from pymavlink import mavutil

# -----------------------------
# Drone / MAVLink helpers
# -----------------------------

def set_param(vehicle, name, value, retries=3, wait=0.25):
    """Robust parameter set with a few retries (DroneKit can timeout sometimes)."""
    for attempt in range(1, retries+1):
        with suppress(Exception):
            vehicle.parameters[name] = value
            time.sleep(wait)
            # (Optional) You can read-back to confirm:
            if name in vehicle.parameters and float(vehicle.parameters[name]) == float(value):
                print(f"Param {name} set to {value}")
                return True
        print(f"Warning: failed to set {name} attempt {attempt}/{retries}")
        time.sleep(wait)
    print(f"ERROR: could not set param {name} after {retries} retries")
    return False

def init_vehicle(connection_string='/dev/ttyACM0', baud=57600, bench_disable_failsafes=True):
    print("Connecting to Cube Orange...")
    vehicle = connect(connection_string, baud=baud, wait_ready=True)
    print(f"Connected to: {vehicle.version}")

    if bench_disable_failsafes:
        # Disable failsafes for bench test (NO PROPS)
        # You can comment these if you prefer keeping them enabled.
        set_param(vehicle, 'FS_THR_ENABLE', 0)
        set_param(vehicle, 'FS_BATT_ENABLE', 0)
        set_param(vehicle, 'ARMING_CHECK', 0)
        print("Failsafes disabled for bench testing.")

    # Set mode once
    print("Setting mode to GUIDED_NOGPS...")
    vehicle.mode = VehicleMode("GUIDED_NOGPS")
    t0 = time.time()
    while vehicle.mode.name != "GUIDED_NOGPS":
        if time.time() - t0 > 10:
            raise RuntimeError("Timeout changing to GUIDED_NOGPS")
        print("Waiting for mode change...")
        time.sleep(1)
    print("Mode set to GUIDED_NOGPS")

    # Arm once
    print("Arming motors (no props)...")
    vehicle.armed = True
    t0 = time.time()
    while not vehicle.armed:
        if time.time() - t0 > 15:
            raise RuntimeError("Timeout arming vehicle")
        print(f"Armed: {vehicle.armed}, System: {vehicle.system_status.state}")
        time.sleep(1)
    print("Vehicle armed and ready.")
    return vehicle

def send_body_velocity(vehicle, vx, vy, vz, yaw_rate):
    """
    vx, vy, vz in m/s (drone body frame).
    yaw_rate in rad/s
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000011111000111,  # enable velocity + yaw_rate only
        0, 0, 0,    # x, y, z positions (not used)
        vx, vy, vz, # velocity in m/s
        0, 0, 0,    # acceleration (not used)
        0,          # yaw (not used)
        yaw_rate    # in rad/s
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

# -----------------------------
# YOLO / Detection utilities
# -----------------------------

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, check_requirements, check_imshow, colorstr,
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh,
    strip_optimizer, set_logging, increment_path, save_one_box
)
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# -----------------------------
# Thread targets
# -----------------------------

def mavlink_worker(vehicle, cmd_queue, stop_event, hz=10.0, zero_timeout=0.5):
    """
    Sends the most recent (vx, vy, vz, yaw_rate) at fixed rate.
    If no new command for `zero_timeout` seconds, sends zeros to hold still.
    """
    period = 1.0 / hz
    last_cmd = (0.0, 0.0, 0.0, 0.0)
    last_recv = time.time()

    print("MAVLink worker started.")
    while not stop_event.is_set():
        # Non-blocking get; keep newest only
        try:
            while True:
                last_cmd = cmd_queue.get_nowait()
                last_recv = time.time()
        except queue.Empty:
            pass

        # If stale, zero it
        if (time.time() - last_recv) > zero_timeout:
            cmd = (0.0, 0.0, 0.0, 0.0)
        else:
            cmd = last_cmd

        with suppress(Exception):
            send_body_velocity(vehicle, *cmd)

        # Optional: small print every second for status
        # print(f"[MAVLink] cmd: vx={cmd[0]:.3f}, vy={cmd[1]:.3f}, vz={cmd[2]:.3f}, yaw_rate={cmd[3]:.3f}")

        time.sleep(period)

    # On exit, send zero a few times to ensure idle
    for _ in range(5):
        with suppress(Exception):
            send_body_velocity(vehicle, 0, 0, 0, 0)
        time.sleep(0.05)
    print("MAVLink worker stopped.")

def yolo_worker(
    weights, source, imgsz, conf_thres, iou_thres, max_det,
    device_str, view_img, save_txt, save_conf, save_crop, nosave,
    classes, agnostic_nms, augment, update, project, name, exist_ok,
    line_thickness, hide_labels, hide_conf, half,
    cmd_queue, stop_event
):
    """
    Runs YOLO inference, computes control, pushes commands to cmd_queue,
    and draws UI. Exits when stop_event is set or window is closed/'q' pressed.
    """
    save_img = not nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(device_str)
    half &= device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()

    # Second-stage classifier (disabled)
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Prime model on CUDA
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    print("YOLO worker started.")
    t0 = time.time()
    try:
        for path, img, im0s, vid_cap in dataset:
            if stop_event.is_set():
                break

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_synchronized()

            for i, det in enumerate(pred):  # detections per image
                if webcam:
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
                s += '%gx%g ' % img.shape[2:]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                imc = im0.copy() if save_crop else im0

                frame_h, frame_w = im0.shape[:2]
                cv2.putText(im0, f"Frame: {frame_w}x{frame_h}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    # Only send control for the most confident detection (max_det=1 already limits)
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        x1, y1, x2, y2 = map(int, xyxy)
                        x_center = int((x1 + x2) / 2)
                        y_center = int((y1 + y2) / 2)
                        cv2.putText(im0, f"Center: ({x_center},{y_center})", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print(f"Detected {names[int(cls)]} at (x={x_center}, y={y_center}) with conf={conf:.2f}")
                        cv2.circle(im0, (x_center, y_center), 5, (0, 0, 255), -1)

                        # --------- control logic ---------------------
                        # (0,0) is top-left of frame
                        err_x = x_center - (frame_w // 2)
                        err_y = y_center - (frame_h // 2)
                        bbox_h = y2 - y1
                        desired_size = frame_h / 5
                        err_z = bbox_h - desired_size

                        # Tune these gains
                        kx = 0.001   # yaw rate per pixel error (rad/s/pixel)
                        ky = 0.001   # vertical speed per pixel error (m/s/pixel)
                        kz = 0.002   # forward speed per pixel error (m/s/pixel)

                        vx = -kz * err_z       # forward/backward (pitch)
                        vy = 0.0               # lateral (roll) kept zero here
                        vz = -ky * err_y       # up/down (throttle)
                        yaw_rate = kx * err_x  # rotate left/right

                        # Clamp to safe magnitudes
                        def clamp(v, lo, hi): return max(lo, min(hi, v))
                        vx = clamp(vx, -1.5, 1.5)
                        vz = clamp(vz, -1.0, 1.0)
                        yaw_rate = clamp(yaw_rate, -0.5, 0.5)

                        # Push latest command (non-blocking, drop old if queue is full)
                        with suppress(queue.Full):
                            # keep queue small; newest wins
                            while True:
                                cmd_queue.get_nowait()
                        # Put newest
                        cmd_queue.put_nowait((vx, vy, vz, yaw_rate))
                        # -----------------------------------------------------------

                        if save_img or save_crop or view_img:
                            c = int(cls)
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                         line_thickness=line_thickness)
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' /
                                             names[c] / f'{p.stem}.jpg', BGR=True)

                # FPS, console print
                fps = 1 / max(1e-6, (t2 - t1))
                cv2.putText(im0, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f'{s}Done. ({t2 - t1:.3f}s) FPS: {fps:.2f}')

                if view_img:
                    # UI crosshair
                    h, w, _ = im0.shape
                    cv2.circle(im0, (w // 2, h // 2), 5, (255, 0, 0), -1)
                    cv2.imshow(str(p), im0)
                    # 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break

                # Save results if requested
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()
                            if vid_cap:
                                vfps = vid_cap.get(cv2.CAP_PROP_FPS)
                                vw = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                vh = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:
                                vfps, vw, vh = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path,
                                                         cv2.VideoWriter_fourcc(*'mp4v'),
                                                         vfps, (vw, vh))
                        vid_writer.write(im0)

            if stop_event.is_set():
                break

    finally:
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")
        print(f'Done. ({time.time() - t0:.3f}s)')
        cv2.destroyAllWindows()
        print("YOLO worker stopped.")

# -----------------------------
# Argparse / main
# -----------------------------

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU threshold')
    parser.add_argument('--max-det', type=int, default=1, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
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
    # Connection args (optional)
    parser.add_argument('--conn', type=str, default='/dev/ttyACM0', help='MAVLink serial/UDP/TCP connection string')
    parser.add_argument('--baud', type=int, default=57600, help='Baud for serial connection')
    parser.add_argument('--keep-armed', action='store_true', help='Do not disarm on exit (bench only)')
    return parser.parse_args()

def main():
    opt = parse_opt()
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))

    # Vehicle init (connect, mode, arm)
    try:
        vehicle = init_vehicle(connection_string=opt.conn, baud=opt.baud, bench_disable_failsafes=True)
    except APIException as e:
        print(f"DroneKit/APIException: {e}")
        return
    except Exception as e:
        print(f"Vehicle init failed: {e}")
        return

    # Command queue (latest wins); capacity 1 so we always keep the newest command
    cmd_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    # Start MAVLink worker (10 Hz)
    mav_t = threading.Thread(target=mavlink_worker, args=(vehicle, cmd_queue, stop_event, 10.0, 0.5), daemon=True)
    mav_t.start()

    # Start YOLO worker (producer)
    yolo_args = (
        opt.weights, opt.source, opt.imgsz, opt.conf_thres, opt.iou_thres, opt.max_det,
        opt.device, opt.view_img, opt.save_txt, opt.save_conf, opt.save_crop, opt.nosave,
        opt.classes, opt.agnostic_nms, opt.augment, opt.update, opt.project, opt.name, opt.exist_ok,
        opt.line_thickness, opt.hide_labels, opt.hide_conf, opt.half,
        cmd_queue, stop_event
    )
    yolo_t = threading.Thread(target=yolo_worker, args=yolo_args, daemon=True)
    yolo_t.start()

    # Wait for YOLO worker to end (q pressed or error)
    try:
        while yolo_t.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: requesting stop.")
    finally:
        stop_event.set()
        yolo_t.join(timeout=2.0)
        mav_t.join(timeout=2.0)

        # Zero velocity on exit
        with suppress(Exception):
            for _ in range(5):
                send_body_velocity(vehicle, 0, 0, 0, 0)
                time.sleep(0.05)

        if not opt.keep_armed:
            with suppress(Exception):
                vehicle.armed = False
                time.sleep(1)
        with suppress(Exception):
            vehicle.close()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
