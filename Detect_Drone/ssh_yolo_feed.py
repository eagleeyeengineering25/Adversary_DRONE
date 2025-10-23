import argparse
import socket
import struct
import time
import cv2
import numpy as np
from detect_test_jetty import run as yolo_run  # your existing detect_test_jetty.py
from detect_test_attitude_jetty import (
    run_with_attitude_control,
    process_detection_with_control,
    send_attitude_target
)
from dronekit import connect, VehicleMode

"""
copy commands
scp eagle@10.250.240.81:/home/eagle/Adversary_DRONE/Detect_Drone/ssh_feed.py C:\Users\savya\Downloads\ssh_feed.py  

command to run after activating the virtual environment:
ON JETSON NANO (SERVER SIDE - YOLO only):
python3 /home/eagle/Adversary_DRONE/Detect_Drone/ssh_yolo_feed.py --server --host 0.0.0.0 --port 5000 --device 0 --quality 60 --weights /home/eagle/Adversary_DRONE/Detect_Drone/best.pt

ON JETSON NANO (SERVER SIDE - with Attitude Control):
sudo /home/eagle/venv/bin/python3 /home/eagle/Adversary_DRONE/Detect_Drone/ssh_yolo_feed.py --server --host 0.0.0.0 --port 5000 --device 0 --quality 60 --weights /home/eagle/Adversary_DRONE/Detect_Drone/best.pt --attitude-control --serial /dev/ttyACM0 --baud 57600

ON LOCAL MACHINE (CLIENT SIDE - replace 10.250.240.81 with Jetson's IP):
python3 ssh_yolo_feed.py --client --host 10.250.240.81 --port 5000
"""
"""
If you have Cd into Detect_Drone and need a new tunnnel
# On Jetson (server) - connects to Cube Orange
python3 ssh_yolo_feed.py --server --host 0.0.0.0 --port 5000 --device 0 --quality 60 --weights best.pt --attitude-control --serial /dev/ttyACM0 --baud 57600

# On laptop (client) - through SSH tunnel
ssh -L 5000:localhost:5000 eagle@10.250.240.81 
python3 ssh_yolo_feed.py --client --host 127.0.0.1 --port 5000 (seperate terminal)
"""
DEFAULT_PORT = 5000
HEADER_FMT = "!I"  # 4-byte length header

def serve(host="0.0.0.0", port=DEFAULT_PORT, jpeg_quality=80, device=0, weights='best.pt', 
          attitude_control=False, serial_port='/dev/ttyACM0', baud=57600):
    """Server: runs YOLO and streams frames to client, optionally with attitude control"""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[server] listening on {host}:{port}")
    conn, addr = srv.accept()
    print(f"[server] client connected: {addr}")

    def send_frame(frame):
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            return
        data = jpg.tobytes()
        header = struct.pack(HEADER_FMT, len(data))
        try:
            conn.sendall(header + data)
        except BrokenPipeError:
            print("[server] client disconnected")
            return False
        return True

    vehicle = None
    try:
        if attitude_control:
            # Connect to Cube Orange for attitude control
            print(f"[server] Connecting to Cube Orange on {serial_port}...")
            vehicle = connect(serial_port, baud=baud, wait_ready=True)
            print(f"[server] Connected. Vehicle mode: {vehicle.mode.name}, Armed: {vehicle.armed}")

            # Set mode to GUIDED_NOGPS
            print("[server] Setting mode to GUIDED_NOGPS...")
            vehicle.mode = VehicleMode("GUIDED_NOGPS")
            while vehicle.mode.name != 'GUIDED_NOGPS':
                print("[server] Waiting for mode change...")
                time.sleep(1)
            print("[server] Mode set to GUIDED_NOGPS")

            # Arm motors (NO PROPS for testing!)
            vehicle.armed = True
            while not vehicle.armed:
                print("[server] Waiting for arming...")
                time.sleep(1)
            print("[server] Vehicle armed!")

        # Use detect_test_jetty.run with a frame_callback to stream frames
        def frame_callback(frame, det, names, frame_w, frame_h):
            """
            Callback for each processed frame.
            If attitude_control enabled, process detections and send commands.
            Always stream the frame to the client.
            """
            if attitude_control and vehicle is not None:
                # Process detections with attitude control
                if len(det) > 0:
                    process_detection_with_control(frame, det, names, frame_w, frame_h, vehicle)
                else:
                    print("[server] No target detected. Hovering...")
                    send_attitude_target(vehicle, 0, 0, 0, 0.3)
            
            # Stream frame to client
            return send_frame(frame)

        # Call YOLO run with frame callback
        yolo_run(
            weights=weights,
            source=str(device) if isinstance(device, int) else device,
            max_det=1,
            imgsz=320,
            nosave=True,
            device=str(device),
            half=True,
            view_img=False,
            frame_callback=frame_callback,
            draw=True,  # Enable bounding box drawing
        )
    finally:
        if attitude_control and vehicle is not None:
            # Disarm and close vehicle connection
            print("[server] Disarming vehicle...")
            vehicle.armed = False
            time.sleep(1)
            vehicle.close()
            print("[server] Vehicle connection closed")
        
        conn.close()
        srv.close()
        print("[server] closed")

def recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return bytes(buf)
        buf.extend(chunk)
    return bytes(buf)

def client(host="localhost", port=DEFAULT_PORT, window_name="YOLO Stream"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            header = recvall(sock, struct.calcsize(HEADER_FMT))
            if not header: break
            (length,) = struct.unpack(HEADER_FMT, header)
            data = recvall(sock, length)
            if not data: break
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        sock.close()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--server", action="store_true", help="Run as server (Jetson)")
    group.add_argument("--client", action="store_true", help="Run as client (viewing machine)")
    parser.add_argument("--host", default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port number")
    parser.add_argument("--device", type=int, default=0, help="Camera device (server only)")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality (server only)")
    parser.add_argument("--weights", type=str, default='best.pt', help='Path to model weights (server only)')
    
    # Attitude control options
    parser.add_argument("--attitude-control", action="store_true", 
                        help="Enable attitude control to Cube Orange (server only)")
    parser.add_argument("--serial", type=str, default='/dev/ttyACM0',
                        help="Serial port for Cube Orange (server only)")
    parser.add_argument("--baud", type=int, default=57600,
                        help="Baud rate for serial connection (server only)")
    
    args = parser.parse_args()

    if args.server:
        serve(
            port=args.port, 
            device=args.device, 
            jpeg_quality=args.quality, 
            weights=args.weights,
            attitude_control=args.attitude_control,
            serial_port=args.serial,
            baud=args.baud
        )
    else:
        client(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
