"""
L1_lidar_GUI.py — Real-time GUI visualizer for SICK TiM LiDAR (270°)
Author: Boo Man & GPT-5

Usage:
------
# Choose local connection type via CLI (no tunnel required)
python3 L1_lidar_GUI.py --ethernet   # use Ethernet (default)
python3 L1_lidar_GUI.py --usb        # use USB (via L1_lidar_usb)

Ethernet quick setup:
---------------------
sudo ip addr add 168.254.15.100/16 dev eth0
sudo ip link set eth0 up
ping 168.254.15.1

USB quick notes:
----------------
- No need to specify /dev/tty*: USB handled by L1_lidar_usb (serial or bulk)
- If permissions block access, add a udev rule or run with sudo as a fallback
"""

import socket
import numpy as np
import time
import threading
import collections
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from L1_lidar_usb import Lidar as UsbLidar

# ========== CONFIGURATION ==========
# Ethernet settings
SENSOR_IP = "168.254.15.1"
SENSOR_PORT = 2112

# Angular resolution settings
# TiM561 supports: 0.33° (810 points), 0.5° (540 points), or 1.0° (270 points)
# Options: '0.33', '0.5', or '1.0'
ANGULAR_RESOLUTION = '1.0'  # Default: 0.33° for maximum resolution
# ================================================== #

STX, ETX = b'\x02', b'\x03'


# --- LiDAR Helper Functions --- #
def bytes_from_stream(stream, is_socket=True):
    """Yield bytes from the stream (socket or serial)."""
    while True:
        if is_socket:
            data = stream.recv(256)
        else:
            data = stream.read(256)
        for b in data:
            yield bytes([b])


def datagrams_from_stream(stream, is_socket=True):
    """Generate datagrams starting with STX and ending with ETX."""
    byte_gen = bytes_from_stream(stream, is_socket)
    while True:
        datagram = b''
        # Find STX
        for b in byte_gen:
            if b == STX:
                break
        # Read until ETX
        for b in byte_gen:
            if b == ETX:
                break
            datagram += b
        yield datagram


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


# --- LiDAR Class --- #
class Lidar:
    def __init__(self, mode='ethernet', ip=SENSOR_IP, tcp_port=SENSOR_PORT, angular_resolution=ANGULAR_RESOLUTION):
        self.mode = mode
        self.ip = ip
        self.tcp_port = tcp_port
        self.angular_resolution = angular_resolution
        self.stop_flag = False
        self.ds = None
        self.connection = None  # Will be socket or serial object
        self.datagrams_generator = None
        # USB implementation (delegates to L1_lidar_usb)
        self._usb_impl = None
        self._usb_proc = None

    def _set_angular_resolution(self):
        """Set the angular resolution of the LiDAR."""
        # Map angular resolution to frequency code
        resolution_map = {
            '0.33': '1',  # 0.33° = 15 Hz
            '0.5': '2',   # 0.5° = 25 Hz
            '1.0': '3'    # 1.0° = 50 Hz
        }
        
        freq_code = resolution_map.get(self.angular_resolution, '1')
        cmd = f'\x02sMN mLMPsetscancfg +{freq_code} +1 -450000 +2250000\x03\0'.encode()
        
        if self.mode == 'ethernet':
            self.connection.send(cmd)
        else:
            # USB handled by underlying implementation
            pass
        
        time.sleep(0.2)
        print(f"[+] Set angular resolution to {self.angular_resolution}°")

    def connect(self):
        """Connect to LiDAR via Ethernet (socket) or USB (L1_lidar_usb)."""
        try:
            if self.mode == 'ethernet':
                # Connect via TCP/IP socket
                self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connection.settimeout(5)
                self.connection.connect((self.ip, self.tcp_port))
                print(f"[+] Connected to LiDAR via Ethernet at {self.ip}:{self.tcp_port}")
                
                # Set angular resolution
                self._set_angular_resolution()
                
                # Activate data streaming
                self.connection.send(b'\x02sEN LMDscandata 1\x03\0')
                self.datagrams_generator = datagrams_from_stream(self.connection, is_socket=True)
                
            elif self.mode == 'usb':
                # Delegate USB to shared implementation (serial or USB bulk)
                self._usb_impl = UsbLidar(angular_resolution=self.angular_resolution)
                self._usb_impl.connect()
                print("[+] Connected to LiDAR via USB (L1_lidar_usb)")
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Use 'ethernet' or 'usb'.")
                
        except Exception as e:
            print(f"[!] Connection error: {e}")

    def run(self):
        """Read continuous data from LiDAR."""
        print("[*] Starting LiDAR stream...")
        if self.mode == 'ethernet':
            while not self.stop_flag:
                try:
                    datagram = next(self.datagrams_generator)
                    decoded = decode_datagram(datagram)
                    if decoded is not None:
                        self.ds = decoded
                except Exception:
                    time.sleep(0.01)
        else:
            # Start underlying USB reader and bridge ds
            self._usb_proc = self._usb_impl.run()
            while not self.stop_flag:
                try:
                    if self._usb_impl.ds is not None:
                        self.ds = self._usb_impl.ds
                    time.sleep(0.02)
                except Exception:
                    time.sleep(0.02)
        print("[*] LiDAR stream stopped")

    def stop(self):
        """Stop LiDAR."""
        self.stop_flag = True
        if self.mode == 'ethernet':
            if self.connection:
                try:
                    self.connection.close()
                except Exception:
                    pass
        else:
            try:
                if self._usb_impl and self._usb_proc:
                    self._usb_impl.kill(self._usb_proc)
            except Exception:
                pass


# --- GUI Class --- #
class LidarGUI:
    def __init__(self, lidar: Lidar):
        self.lidar = lidar
        self.root = tk.Tk()
        self.root.title("SICK TiM561 LiDAR Visualizer (270°)")

        # Matplotlib figure
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, polar=True)
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_thetalim(-np.pi * 3 / 4, np.pi * 3 / 4)
        self.ax.set_title("Live LiDAR Scan", va='bottom')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.scatter = None
        self.update_plot()

    def update_plot(self):
        """Update polar plot with latest LiDAR scan."""
        if self.lidar.ds is not None:
            angles = np.linspace(-135, 135, len(self.lidar.ds)) * np.pi / 180
            distances = self.lidar.ds

            self.ax.clear()
            self.ax.set_theta_zero_location('N')
            self.ax.set_theta_direction(-1)
            self.ax.set_thetalim(-np.pi * 3 / 4, np.pi * 3 / 4)
            self.ax.scatter(angles, distances, s=5, c='cyan')
            self.ax.set_title("SICK TiM561 Live 270° Scan", va='bottom')
            self.canvas.draw()

        self.root.after(100, self.update_plot)  # refresh every 100ms

    def run(self):
        """Run the GUI main loop."""
        self.root.mainloop()


# --- Main Execution --- #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Local LiDAR GUI over Ethernet or USB")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ethernet", action="store_true", help="use Ethernet (default)")
    group.add_argument("--usb", action="store_true", help="use USB via L1_lidar_usb")
    parser.add_argument("--ip", default=SENSOR_IP, help="sensor IP for Ethernet mode")
    parser.add_argument("--port", type=int, default=SENSOR_PORT, help="sensor TCP port for Ethernet mode")
    parser.add_argument("--resolution", choices=["0.33","0.5","1.0"], default=ANGULAR_RESOLUTION, help="angular resolution")
    args = parser.parse_args()

    mode = 'usb' if args.usb else 'ethernet'
    lidar = Lidar(mode=mode, ip=args.ip, tcp_port=args.port, angular_resolution=args.resolution)
    lidar.connect()

    t = threading.Thread(target=lidar.run, daemon=True)
    t.start()

    gui = LidarGUI(lidar)
    try:
        gui.run()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        lidar.stop()
