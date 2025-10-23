import argparse
import socket
import struct
import sys
import threading
import time
import cv2
import numpy as np

#!/usr/bin/env python3
r"""
ssh_feed.py
#Copy the file on local only needed once (step1 for both ssh_feed.py and ssh_yolo_feed.py)
scp eagle@10.250.240.81:/home/eagle/Adversary_DRONE/Detect_Drone/ssh_feed.py C:\Users\savya\Downloads\ssh_feed.py  

Run on the companion (camera-connected) as a server:
    python3 ssh_feed.py --server --port 5000

On the laptop, forward the port over SSH and run the client:
    # on laptop
    make sure to donwload the ssh_feed file locally with python and its dependencies built
    ssh -L 5000:localhost:5000 eagle@10.250.240.81 &
    python3 ssh_feed.py --client --host localhost --port 5000

    #optionally, if you have ssh key setup and want only one active terminal window use:
    Start-Job { ssh -N -L 5000:127.0.0.1:5000 eagle@10.250.240.81 } ; cd "$env:USERPROFILE\Downloads" ; python ssh_feed.py --client --host 127.0.0.1 --port 5000
    
    #run following commands to setup key 
    ssh-keygen -t ed25519
    type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh eagle@10.250.240.81 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"



This script sends JPEG-encoded frames over TCP (length-prefixed). Use SSH local port forwarding
to tunnel the TCP stream securely.
"""



DEFAULT_PORT = 5000
HEADER_FMT = "!I"  # 4-byte unsigned int, big-endian


def serve(port: int, device: int = 0, jpeg_quality: int = 80):
        addr = ("0.0.0.0", port)
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(addr)
        srv.listen(1)
        print(f"[server] listening on {addr}")

        try:
                while True:
                        conn, client = srv.accept()
                        print(f"[server] client connected: {client}")
                        try:
                                cap = cv2.VideoCapture(device)
                                if not cap.isOpened():
                                        print("[server] failed to open camera")
                                        conn.close()
                                        break

                                # Send frames until client disconnects
                                while True:
                                        ret, frame = cap.read()
                                        if not ret:
                                                print("[server] frame read failed")
                                                break

                                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                                        ok, jpg = cv2.imencode(".jpg", frame, encode_param)
                                        if not ok:
                                                continue
                                        data = jpg.tobytes()
                                        header = struct.pack(HEADER_FMT, len(data))
                                        # send header + data
                                        try:
                                                conn.sendall(header)
                                                conn.sendall(data)
                                        except BrokenPipeError:
                                                print("[server] client disconnected")
                                                break

                                        # optional throttle
                                        time.sleep(0.01)
                        finally:
                                try:
                                        conn.close()
                                except Exception:
                                        pass
                                try:
                                        cap.release()
                                except Exception:
                                        pass
                                print("[server] connection closed, waiting for next client")
        except KeyboardInterrupt:
                print("[server] stopped by user")
        finally:
                srv.close()


def recvall(sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
                chunk = sock.recv(n - len(buf))
                if not chunk:
                        return bytes(buf)
                buf.extend(chunk)
        return bytes(buf)


def client(host: str, port: int, window_name: str = "remote"):
        addr = (host, port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[client] connecting to {addr} ...")
        sock.connect(addr)
        print("[client] connected")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
                while True:
                        header = recvall(sock, struct.calcsize(HEADER_FMT))
                        if not header or len(header) < struct.calcsize(HEADER_FMT):
                                print("[client] no header (server closed?)")
                                break
                        (length,) = struct.unpack(HEADER_FMT, header)
                        data = recvall(sock, length)
                        if not data or len(data) < length:
                                print("[client] incomplete frame received")
                                break
                        arr = np.frombuffer(data, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is None:
                                continue
                        cv2.imshow(window_name, frame)
                        # Exit on ESC or 'q'
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27 or key == ord("q"):
                                break
        except KeyboardInterrupt:
                pass
        finally:
                sock.close()
                cv2.destroyAllWindows()
                print("[client] closed")


def main():
        parser = argparse.ArgumentParser(description="Stream camera frames over TCP (use SSH port forwarding).")
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--server", action="store_true", help="run in server mode (camera attached here)")
        group.add_argument("--client", action="store_true", help="run in client mode (display on this machine)")
        parser.add_argument("--host", default="localhost", help="server host for client mode")
        parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to use")
        parser.add_argument("--device", type=int, default=0, help="camera device index (server)")
        parser.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100 (server)")
        args = parser.parse_args()

        if args.server:
                serve(port=args.port, device=args.device, jpeg_quality=args.quality)
        else:
                client(host=args.host, port=args.port)


if __name__ == "__main__":
        main()