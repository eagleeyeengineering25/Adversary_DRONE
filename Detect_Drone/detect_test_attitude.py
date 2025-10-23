"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ Example: python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
    
    Activate the virtual environment first:
    $ source venv/bin/activate
    
    $ cd Adversary_DRONE/Detect_Drone/
    $ python detect_test_velocity.py --weights best.pt --source 0 --max-det 1 --imgsz 320 --nosave
    
    THIS SCRIPT tests sending velocity commands
    
    1. YOLO detects target drone -> gets bounding box center (x_center, y_center)
    2. computes error (err_x, err_y) relative to frame center
    3. map error -> velocity + yaw rate
    4. sends MAVLink messgeas via Pi to Cube Orange
    5. Cube Orange adjusts attitude/yaw -> keeps target drone cenetered
    6. Mission Planner shows live response
    
"""

# added to bypass UnpicklingError: Weights only load failed
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

# ADDED IMPORTS
from dronekit import connect, VehicleMode
from pymavlink import mavutil

import math

def to_quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    """Convert Euler angles (radians) to quaternion [w, x, y, z]."""
    t0 = math.cos(yaw * 0.5)
    t1 = math.sin(yaw * 0.5)
    t2 = math.cos(roll * 0.5)
    t3 = math.sin(roll * 0.5)
    t4 = math.cos(pitch * 0.5)
    t5 = math.sin(pitch * 0.5)
    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5
    return [w, x, y, z]


print("Connecting to Cube Orange...")
vehicle = connect('/dev/ttyACM0', baud=57600, wait_ready=True)
"""
# check PORT & BAUD for telemetry/serial connection
# /dev/serial0 is default UART on the Pi GPIO pins (TX/RX)
# if conneted by USB, found /dev/ttyACM0 and /dev/ttyACM1
"""

# Set mode to GUIDED_NOGPS
print("Setting mode to GUIDED_NOGPS...")
vehicle.mode = VehicleMode("GUIDED_NOGPS")
while not vehicle.mode.name == 'GUIDED_NOGPS':
    print("Waiting for mode change...")
    time.sleep(1)
print("Mode set to GUIDED_NOGPS")

# # Set mode to GUIDED
# print("Setting mode to GUIDED...")
# vehicle.mode = VehicleMode("GUIDED")
# while not vehicle.mode.name == 'GUIDED':
#     print("Waiting for mode change...")
#     time.sleep(1)
# print("Mode set to GUIDED")

# Arm motors (NO PROPS!)
vehicle.armed = True
while not vehicle.armed:
    print("Waiting for arming...")
    time.sleep(1)
print("Vehicle armed!")

while not vehicle.armed:
    print("Waiting for vehicle to arm...")
    time.sleep(1)
    
    print("Vehicle armed and in GUIDED mode.")
    
# --- Optional sanity check ---
print(f"Vehicle mode: {vehicle.mode.name}, Armed: {vehicle.armed}")


FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# SEND VELOCITY/YAW
def send_attitude_target(vehicle, roll=0.0, pitch=0.0, yaw_rate=0.0, thrust=0.0):
    """
    Send SET_ATTITUDE_TARGET MAVLink message.
    roll, pitch in radians
    yaw_rate in rad/s
    thrust: 0.0 - 1.0
    """
    q = to_quaternion(roll, pitch, 0)  # yaw=0 for now

    msg = vehicle.message_factory.set_attitude_target_encode(
        0,  # time_boot_ms
        1,  # target system
        1,  # target component
        0b00000111,  # ignore body rates except yaw_rate
        q,           # quaternion [w, x, y, z]
        0, 0, yaw_rate,  # roll_rate, pitch_rate, yaw_rate
        thrust            # thrust 0-1
    )

    vehicle.send_mavlink(msg)
    vehicle.flush()

    # optional debug print
    print(f"Sent attitude target: roll={roll:.3f}, pitch={pitch:.3f}, yaw_rate={yaw_rate:.3f}, thrust={thrust:.3f}")

    

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not in crement
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        
        frame_callback: Optional[Callable] = None,  # optional per-frame callback(frame) -> bool/None
	    draw: bool = True,  # whether to draw boxes/labels on frames
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    # set_logging()
    # device = select_device(device)
    # half &= device.type != 'cpu'  # half precision only supported on CUDA
    set_logging()
    force_cpu = False  # Set this to True to force CPU usage
    device = torch.device('cpu') if force_cpu else select_device(device)
    print(f'Using device: {device}, CUDA available: {not force_cpu and torch.cuda.is_available()}')
    if not force_cpu and torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name()}')
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            # Get frame height and width and display on frame
            frame_h, frame_w = im0.shape[:2]
            cv2.putText(im0, f"Frame: {frame_w}x{frame_h}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # get bounding box corners, center, and print center coordinates
                    x1, y1, x2, y2 = map(int, xyxy)  # bounding box corners
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)
                    
                    # Display center coordinates on frame
                    cv2.putText(im0, f"Center: ({x_center},{y_center})", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    print(f"Detected {names[int(cls)]} at (x={x_center}, y={y_center}) with conf={conf:.2f}")
                    cv2.circle(im0, (x_center, y_center), 5, (0, 0, 255), -1)  # mark center

 
                    # --------- Control logic ---------------------
                    """
                    (0,0) is top-left of frame
                    The drone will:
                    - Yaw left/right to center the target horizontally (x-axis)
                    - Pitch forward/back to control distance (z-axis)
                    - Adjust thrust for vertical alignment (y-axis)
                    - Keep roll = 0 for stability
                    """
                    err_x = x_center - (frame_w // 2) # gets difference from center points in x axis (pixels)
                    err_y = y_center - (frame_h // 2) # gets difference from center points in y axis (pixels)
                    
                    # size-based depth estimation
                    bbox_h = y2 - y1                
                    desired_size = frame_h / 5      
                    err_z = bbox_h - desired_size 

                    # Tunable gains
                    Kp_pitch = 0.001   # forward/backward
                    Kp_thrust = 0.0005 # up/down
                    Kp_yaw = 0.001     # yaw rate (horizontal centering)

                    # Convert image-space errors to attitude commands
                    pitch = -Kp_pitch * err_z               # forward/backward tilt
                    roll = 0.0
                    yaw_rate = Kp_yaw * err_x               # turning left/right
                    thrust = 0.25 + (Kp_thrust * (-err_y))  # small correction for altitude

                    # --- Clip values to safe ranges ---
                    pitch    = np.clip(pitch, -0.3, 0.3)
                    yaw_rate = np.clip(yaw_rate, -0.5, 0.5)
                    thrust   = np.clip(thrust, 0.2, 0.6)
                    # pitch    = max(-0.3, min(0.3, pitch))
                    # yaw_rate = max(-0.5, min(0.5, yaw_rate))
                    # thrust   = max(0.2, min(0.6, thrust))

                    # prints drone movement direction
                    direction = []
                    # Forward/backward (pitch)
                    if pitch > 0.05:
                        direction.append("pitching forward")
                    elif pitch < -0.05:
                        direction.append("pitching backward")

                    # Up/down (thrust)
                    if thrust > 0.45:
                        direction.append("ascending")
                    elif thrust < 0.25:
                        direction.append("descending")

                    # Yaw left/right (rotation)
                    if yaw_rate > 0.05:
                        direction.append("yawing right")
                    elif yaw_rate < -0.05:
                        direction.append("yawing left")
                    action_text = "Drone " + " + ".join(direction) if direction else "Drone holding attitude"
                    # Ex: Drone moving yawing right + pitching forward
                    print(f"{action_text} | pitch={math.degrees(pitch):.1f}°, thrust={thrust:.2f}, yaw_rate={yaw_rate:.2f} rad/s")
                    cv2.putText(im0, action_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                

                    # Send to cube (with short command rate limit, commands are only sent ever 0.1s/100ms)
                    last_send_time = 0
                    current_time = time.time()
                    if current_time - last_send_time >= 0.1:  # send every 0.1 seconds
                        send_attitude_target(vehicle, roll, pitch, yaw_rate, thrust)
                        last_send_time = current_time
                    #-----------------------------------------------------------

                    if save_img or save_crop or view_img:
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            else:
                # hover when no target detected
                print("No target detected. Hovering...")
                send_attitude_target(vehicle, 0, 0, 0, 0.3)
                #time.sleep(0.1)

            # Print time (inference + NMS)
            #print(f'{s}Done. ({t2 - t1:.3f}s)')

            # ***Calculate FPS and display on frame
            fps = 1 / (t2 - t1)
            cv2.putText(im0, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Display center coordinates on frame
            #cv2.putText(im0, f"Center: ({x_center},{y_center})", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Print time (inference + NMS)
            if frame_callback is None:
                print(f'{s}Done. ({t2 - t1:.3f}s) FPS: {fps:.2f}')
            
            # ***also print FPS in terminal
            #print(f"FPS: {fps:.2f}")

            # Stream results
            if frame_callback is not None:
                try:
                    cont = frame_callback(im0)
                    if cont is False:
                        break
                except Exception as e:
                    print(f"frame callback error: {e}")
                    break
            elif view_img:
                # ***add center marker
                h, w, _ = im0.shape
                cv2.circle(im0, (w//2, h//2), 5, (255, 0, 0), -1)
                
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) & 0xFF == ord('q'): # q to quit
                    break

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


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
