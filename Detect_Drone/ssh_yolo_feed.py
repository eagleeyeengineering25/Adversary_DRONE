import argparse
import socket
import struct
import time
import cv2
import numpy as np
from detect_test_jetty import run as yolo_run  # your existing detect_test_jetty.py
"""
command to run after activating the virtual environment:
ON JETSON NANO (SERVER SIDE):
python3 /home/eagle/Adversary_DRONE/Detect_Drone/ssh_yolo_feed.py --server --host 127.0.0.1 --port 5000 --device 0 --quality 60 --weights /home/eagle/Adversary_DRONE/Detect_Drone/best.pt
ON LOCAL MACHINE (CLIENT SIDE):
python3 ssh_yolo_feed.py --client --host 127.0.0.1 --port 5000
"""
DEFAULT_PORT = 5000
HEADER_FMT = "!I"  # 4-byte length header

def serve(host="0.0.0.0", port=DEFAULT_PORT, jpeg_quality=80, device=0, weights='best.pt'):
    """Server: runs YOLO and streams frames to client"""
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

    try:
        # Use detect_test.run with a frame_callback to stream frames without local display/drawing
        def frame_callback(frame):
            # frame is a BGR numpy array (im0). Send as JPEG over socket.
            return send_frame(frame)

        # Call YOLO run with drawing disabled (draw=False) to minimize overhead and use half precision if available
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
    group.add_argument("--server", action="store_true")
    group.add_argument("--client", action="store_true")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--weights", type=str, default='best.pt', help='path to model weights on server')
    args = parser.parse_args()

    if args.server:
        serve(port=args.port, device=args.device, jpeg_quality=args.quality, weights=args.weights)
    else:
        client(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
