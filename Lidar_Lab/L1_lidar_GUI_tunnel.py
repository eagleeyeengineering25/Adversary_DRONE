"""
L1_lidar_GUI_tunnel.py — LiDAR GUI with SSH tunnel support for obstacle avoidance
Author: Boo Man & GPT-5

This combines the working USB LiDAR protocol (via L1_lidar_usb.Lidar) with SSH tunneling for remote visualization.

Usage:
------
# On the Jetson (server with LiDAR connected via USB):
sudo python3 L1_lidar_GUI_tunnel.py --server --port 5100

# On the laptop (client):
# 1. Forward the port over SSH (one-time setup):
ssh -L 5100:localhost:5100 eagle@10.250.240.81 &

# 2. Run client to see GUI remotely:
python3 L1_lidar_GUI_tunnel.py --client --host localhost --port 5100

# Optional: PowerShell one-liner (Windows):
Start-Job { ssh -N -L 5100:127.0.0.1:5100 eagle@10.250.240.81 } ; cd "$env:USERPROFILE\Downloads" ; python L1_lidar_GUI_tunnel.py --client --host 127.0.0.1 --port 5100

Setup SSH key (one-time):
--------------------------
ssh-keygen -t ed25519
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh eagle@10.250.240.81 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
"""

import argparse
import socket
import struct
from L1_lidar_usb import Lidar
import numpy as np
import time
import threading
import pickle
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ========== CONFIGURATION ========== #
USB_VENDOR_ID = 0x19a2
USB_PRODUCT_ID = 0x5001
ANGULAR_RESOLUTION = '1.0'  # '0.33', '0.5', or '1.0'
DEFAULT_PORT = 5100
HEADER_FMT = "!I"  # 4-byte unsigned int, big-endian
# =================================== #

STX, ETX = b'\x02', b'\x03'


## USB connection is handled by L1_lidar_usb.Lidar


# ========== LIDAR PARSING ========== #
def parse_number(n):
    """Parse decimal or hex numbers from bytes."""
    try:
        return int(n, 16)
    except ValueError:
        return int(n)


def decode_datagram(datagram):
    """Extract scan data from the LiDAR datagram."""
    try:
        items = datagram.split(b' ')
        if items[0] != b'sSN' or items[1] != b'LMDscandata':
            return None

        num_data = parse_number(items[25])
        data = [parse_number(x) / 1000 for x in items[26:26 + num_data]]
        return np.array(data)
    except Exception:
        return None


# ========== SERVER (Jetson with LiDAR) ========== #
def serve(port: int):
    """Server mode: Read LiDAR and stream scan data over TCP"""
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
                # Use shared Lidar implementation (auto USB serial/bulk)
                lidar = Lidar(angular_resolution=ANGULAR_RESOLUTION)
                lidar.connect()
                proc = lidar.run()
                print("[*] Streaming LiDAR data to client...")

                while True:
                    ds = lidar.ds
                    if ds is not None:
                        scan = np.array(ds)
                        scan_bytes = pickle.dumps(scan)
                        header = struct.pack(HEADER_FMT, len(scan_bytes))
                        try:
                            conn.sendall(header)
                            conn.sendall(scan_bytes)
                        except BrokenPipeError:
                            print("[server] client disconnected")
                            raise
                    time.sleep(0.02)  # ~50 Hz pacing

            except Exception as e:
                print(f"[server] error: {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                try:
                    lidar.kill(proc)
                except Exception:
                    pass
                print("[server] connection closed, waiting for next client")
                
    except KeyboardInterrupt:
        print("[server] stopped by user")
    finally:
        srv.close()


# ========== CLIENT (Laptop with GUI) ========== #
def recvall(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket"""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return bytes(buf)
        buf.extend(chunk)
    return bytes(buf)


class RemoteLidarGUI:
    """GUI for displaying remote LiDAR scans"""
    def __init__(self, sock):
        self.sock = sock
        self.root = tk.Tk()
        self.root.title("Remote SICK TiM561 LiDAR - Obstacle Avoidance")
        
        # Matplotlib figure
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111, polar=True)
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_thetalim(-np.pi * 3 / 4, np.pi * 3 / 4)
        self.ax.set_ylim(0, 10)  # 10m max range
        self.ax.set_title("Remote LiDAR 270° Scan (SSH Tunnel)", va='bottom', fontsize=14, fontweight='bold')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Waiting for data...", 
                                     font=("Arial", 12), bg="yellow", fg="black")
        self.status_label.pack(fill=tk.X, pady=5)
        
        self.latest_scan = None
        self.running = True
        
        # Start receiver thread
        self.receiver_thread = threading.Thread(target=self.receive_scans, daemon=True)
        self.receiver_thread.start()
        
        # Start update loop
        self.update_plot()
    
    def receive_scans(self):
        """Receive scan data from server"""
        try:
            while self.running:
                # Read header
                header = recvall(self.sock, struct.calcsize(HEADER_FMT))
                if not header or len(header) < struct.calcsize(HEADER_FMT):
                    print("[client] no header (server closed?)")
                    break
                
                (length,) = struct.unpack(HEADER_FMT, header)
                
                # Read scan data
                data = recvall(self.sock, length)
                if not data or len(data) < length:
                    print("[client] incomplete scan received")
                    break
                
                # Deserialize
                scan = pickle.loads(data)
                self.latest_scan = scan
                
        except Exception as e:
            print(f"[client] receiver error: {e}")
        finally:
            self.running = False
    
    def update_plot(self):
        """Update polar plot with latest LiDAR scan"""
        if self.latest_scan is not None:
            distances = self.latest_scan
            angles = np.linspace(-135, 135, len(distances)) * np.pi / 180
            
            # Calculate obstacle warnings
            min_dist = np.min(distances)
            close_obstacles = np.sum(distances < 2.0)  # Objects within 2m
            
            # Update status
            if min_dist < 0.5:
                status = f"⚠️ DANGER! Obstacle at {min_dist:.2f}m"
                status_color = "red"
            elif min_dist < 1.0:
                status = f"⚠️ WARNING: Close obstacle at {min_dist:.2f}m"
                status_color = "orange"
            elif min_dist < 2.0:
                status = f"⚠️ CAUTION: Obstacle at {min_dist:.2f}m ({close_obstacles} points < 2m)"
                status_color = "yellow"
            else:
                status = f"✓ Clear - Nearest: {min_dist:.2f}m"
                status_color = "green"
            
            self.status_label.config(text=status, bg=status_color)
            
            # Color points based on distance (red=close, green=far)
            colors = np.where(distances < 1.0, 'red',
                             np.where(distances < 2.0, 'orange', 'cyan'))
            
            # Clear and redraw
            self.ax.clear()
            self.ax.set_theta_zero_location('N')
            self.ax.set_theta_direction(-1)
            self.ax.set_thetalim(-np.pi * 3 / 4, np.pi * 3 / 4)
            self.ax.set_ylim(0, 10)
            self.ax.scatter(angles, distances, s=10, c=colors, alpha=0.8)
            self.ax.set_title("Remote LiDAR 270° Scan (SSH Tunnel)", 
                            va='bottom', fontsize=14, fontweight='bold')
            self.canvas.draw()
        
        if self.running:
            self.root.after(50, self.update_plot)  # 20 Hz refresh
        else:
            self.root.quit()
    
    def run(self):
        """Run the GUI main loop"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window close"""
        self.running = False
        self.sock.close()
        self.root.destroy()


def client(host: str, port: int):
    """Client mode: Connect to server and display GUI"""
    addr = (host, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[client] connecting to {addr}...")
    sock.connect(addr)
    print("[client] connected - starting GUI")
    
    try:
        gui = RemoteLidarGUI(sock)
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print("[client] closed")


# ========== MAIN ========== #
def main():
    parser = argparse.ArgumentParser(
        description="Stream LiDAR scans over TCP for remote obstacle avoidance GUI (use SSH port forwarding)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--server", action="store_true", 
                      help="run in server mode (Jetson with LiDAR attached)")
    group.add_argument("--client", action="store_true", 
                      help="run in client mode (laptop to display GUI)")
    parser.add_argument("--host", default="localhost", 
                       help="server host for client mode")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, 
                       help="port to use (default: 5100)")
    args = parser.parse_args()
    
    if args.server:
        serve(port=args.port)
    else:
        client(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
