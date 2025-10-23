"""
L1_lidar_GUI_client.py — Client-only GUI for remote LiDAR via SSH tunnel
Author: Boo Man & GPT-5

This is a lightweight client that connects to the Jetson server running
L1_lidar_GUI_tunnel.py --server and displays the LiDAR scan.

Usage (on laptop):
  1) Create SSH tunnel to Jetson (adjust IP):
       ssh -L 5100:localhost:5100 eagle@<jetson-ip>
  2) Run client:
       python3 L1_lidar_GUI_client.py --host localhost --port 5100

No USB or SICK drivers are needed on the laptop. It only needs Python with
numpy, tkinter, and matplotlib.
"""

import argparse
import socket
import struct
import pickle
import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

DEFAULT_PORT = 5100
HEADER_FMT = "!I"  # 4-byte unsigned int, big-endian


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
        self.root.title("Remote SICK TiM LiDAR - Obstacle Avoidance")

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
        import threading
        self.receiver_thread = threading.Thread(target=self.receive_scans, daemon=True)
        self.receiver_thread.start()

        # Start update loop
        self.update_plot()

    def receive_scans(self):
        """Receive scan data from server"""
        try:
            while self.running:
                header = recvall(self.sock, struct.calcsize(HEADER_FMT))
                if not header or len(header) < struct.calcsize(HEADER_FMT):
                    print("[client] no header (server closed?)")
                    break
                (length,) = struct.unpack(HEADER_FMT, header)

                data = recvall(self.sock, length)
                if not data or len(data) < length:
                    print("[client] incomplete scan received")
                    break

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
            close_obstacles = np.sum(distances < 2.0)

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
            self.root.after(50, self.update_plot)  # ~20 Hz refresh
        else:
            self.root.quit()

    def run(self):
        """Run the GUI main loop"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Handle window close"""
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Client-only GUI for remote LiDAR over SSH tunnel")
    parser.add_argument("--host", default="localhost", help="server host (use localhost with SSH -L)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="server port (default: 5100)")
    args = parser.parse_args()

    addr = (args.host, args.port)
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
        try:
            sock.close()
        except Exception:
            pass
        print("[client] closed")


if __name__ == "__main__":
    main()
