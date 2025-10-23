"""
Jetson CUDA YOLOv5 with Attitude Control for Cube Orange

This script runs YOLOv5 on Jetson with CUDA, detects the target drone,
computes bounding box errors, and sends attitude commands to Cube Orange
to center the target in the frame.

Usage:
    Activate virtual environment first:
    $ cd Adversary_DRONE/
    $ source venv/bin/activate
    
    Run on Jetson:
    $ cd Detect_Drone/
    $ python detect_test_attitude_jetty.py --weights best.pt --source 0 --max-det 1 --imgsz 320 --nosave --device 0
    
Flow:
    1. YOLO detects target drone on Jetson GPU -> gets bounding box center (x_center, y_center)
    2. Computes error (err_x, err_y, err_z) relative to frame center
    3. Maps error -> attitude commands (roll, pitch, yaw_rate, thrust)
    4. Sends MAVLink messages via serial to Cube Orange
    5. Cube Orange adjusts attitude/yaw to keep target centered
"""

import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import argparse
import time
import math
import cv2
import numpy as np

# MAVLink / DroneKit imports
from dronekit import connect, VehicleMode
from pymavlink import mavutil

# Import YOLO detection from detect_test_jetty
from detect_test_jetty import run as yolo_run


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


def send_attitude_target(vehicle, roll=0.0, pitch=0.0, yaw_rate=0.0, thrust=0.0):
    """
    Send SET_ATTITUDE_TARGET MAVLink message.
    roll, pitch in radians
    yaw_rate in rad/s
    thrust: 0.0 - 1.0
    """
    q = to_quaternion(roll, pitch, 0)  # yaw=0 for now

    msg = vehicle.message_factory.set_attitude_target_encode(
        0,              # time_boot_ms
        1,              # target system
        1,              # target component
        0b00000111,     # ignore body rates except yaw_rate
        q,              # quaternion [w, x, y, z]
        0, 0, yaw_rate, # roll_rate, pitch_rate, yaw_rate
        thrust          # thrust 0-1
    )

    vehicle.send_mavlink(msg)
    vehicle.flush()

    print(f"Sent attitude target: roll={roll:.3f}, pitch={pitch:.3f}, yaw_rate={yaw_rate:.3f}, thrust={thrust:.3f}")


def process_detection_with_control(im0, det, names, frame_w, frame_h, vehicle):
    """
    Process YOLO detections and send attitude commands to Cube Orange.
    Called as a callback for each detected bounding box.
    """
    for *xyxy, conf, cls in reversed(det):
        # Get bounding box center
        x1, y1, x2, y2 = map(int, xyxy)
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)

        # Display center coordinates
        cv2.putText(im0, f"Center: ({x_center},{y_center})", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        print(f"Detected {names[int(cls)]} at (x={x_center}, y={y_center}) with conf={conf:.2f}")
        
        # Mark center with color based on confidence
        dot_color = (0, 165, 255) if conf < 0.75 else (0, 0, 255)
        cv2.circle(im0, (x_center, y_center), 5, dot_color, -1)

        # --------- Control logic ---------------------
        """
        (0,0) is top-left of frame
        The drone will:
        - Yaw left/right to center the target horizontally (x-axis)
        - Pitch forward/back to control distance (z-axis)
        - Adjust thrust for vertical alignment (y-axis)
        - Keep roll = 0 for stability
        """
        err_x = x_center - (frame_w // 2)  # horizontal error (pixels)
        err_y = y_center - (frame_h // 2)  # vertical error (pixels)

        # Size-based depth estimation
        bbox_h = y2 - y1
        desired_size = frame_h / 5
        err_z = bbox_h - desired_size

        # Tunable gains
        Kp_pitch = 0.001    # forward/backward
        Kp_thrust = 0.0005  # up/down
        Kp_yaw = 0.001      # yaw rate (horizontal centering)

        # Convert image-space errors to attitude commands
        pitch = -Kp_pitch * err_z           # forward/backward tilt
        roll = 0.0
        yaw_rate = Kp_yaw * err_x           # turning left/right
        thrust = 0.25 + (Kp_thrust * (-err_y))  # altitude correction

        # Clip values to safe ranges
        pitch = np.clip(pitch, -0.3, 0.3)
        yaw_rate = np.clip(yaw_rate, -0.5, 0.5)
        thrust = np.clip(thrust, 0.2, 0.6)

        # Print drone movement direction
        direction = []
        if pitch > 0.05:
            direction.append("pitching forward")
        elif pitch < -0.05:
            direction.append("pitching backward")

        if thrust > 0.45:
            direction.append("ascending")
        elif thrust < 0.25:
            direction.append("descending")

        if yaw_rate > 0.05:
            direction.append("yawing right")
        elif yaw_rate < -0.05:
            direction.append("yawing left")

        action_text = "Drone " + " + ".join(direction) if direction else "Drone holding attitude"
        print(f"{action_text} | pitch={math.degrees(pitch):.1f}°, thrust={thrust:.2f}, yaw_rate={yaw_rate:.2f} rad/s")
        cv2.putText(im0, action_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Send to cube (with short command rate limit, commands are only sent ever 0.1s/100ms)
        last_send_time = 0
        current_time = time.time()
        if current_time - last_send_time >= 0.1:  # send every 0.1 seconds
            send_attitude_target(vehicle, roll, pitch, yaw_rate, thrust)
            last_send_time = current_time
        break


def run_with_attitude_control(
    weights='yolov5s.pt',
    source='0',
    imgsz=640,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device='',
    serial_port='/dev/ttyACM0',
    baud=57600,
    **kwargs
):
    """
    Run YOLO detection with attitude control for Cube Orange.
    """
    # Connect to Cube Orange
    print(f"Connecting to Cube Orange on {serial_port}...")
    vehicle = connect(serial_port, baud=baud, wait_ready=True)
    print(f"Connected. Vehicle mode: {vehicle.mode.name}, Armed: {vehicle.armed}")

    # Set mode to GUIDED_NOGPS
    print("Setting mode to GUIDED_NOGPS...")
    vehicle.mode = VehicleMode("GUIDED_NOGPS")
    while vehicle.mode.name != 'GUIDED_NOGPS':
        print("Waiting for mode change...")
        time.sleep(1)
    print("Mode set to GUIDED_NOGPS")

    # Arm motors (NO PROPS for testing!)
    vehicle.armed = True
    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)
    print("Vehicle armed!")

    # Define frame callback that processes detections with attitude control
    def frame_callback(frame, det, names, frame_w, frame_h):
        """
        Callback invoked by yolo_run for each frame with detection data.
        frame: processed frame with annotations
        det: detection tensor [xyxy, conf, cls]
        names: class names dictionary
        frame_w, frame_h: frame dimensions
        """
        if len(det) > 0:
            # Process detections with attitude control
            process_detection_with_control(frame, det, names, frame_w, frame_h, vehicle)
        else:
            # Hover when no target detected
            print("No target detected. Hovering...")
            send_attitude_target(vehicle, 0, 0, 0, 0.3)
        
        return True  # Continue processing

    try:
        # Call the YOLO detection function from detect_test_jetty
        yolo_run(
            weights=weights,
            source=source,
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            device=device,
            nosave=True,
            view_img=False,
            frame_callback=frame_callback,
            draw=True,
            **kwargs
        )
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Disarm and close vehicle connection
        print("Disarming vehicle...")
        vehicle.armed = False
        time.sleep(1)
        vehicle.close()
        print("Vehicle connection closed")


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
    parser.add_argument('--serial-port', default='/dev/ttyACM0', help='Cube Orange serial port')
    parser.add_argument('--baud', type=int, default=57600, help='Serial baud rate')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
