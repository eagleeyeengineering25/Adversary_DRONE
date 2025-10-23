"""
L1_lidar_usb.py - USB (serial/CDC-ACM) version of L1_lidar.py for SICK TiM561 LiDAR
Author: Boo Man & GPT-5

Usage:
------
sudo python3 L1_lidar_usb.py

This script mirrors L1_lidar.py but talks SOPAS over the LiDAR's USB virtual
serial port (/dev/ttyACM* or /dev/ttyUSB*). It uses the recommended sequence
for TiM5xx: login -> start measurement -> run -> enable LMDscandata.

Why this change? Your unit enumerates as a serial device; using raw libusb
bulk endpoints prevented streaming and put the device in an error state (LED red).
"""

import glob
import time
import numpy as np
import serial
from threading import Thread
import collections
try:
    import usb.core
    import usb.util
    HAVE_USB = True
except Exception:
    HAVE_USB = False

# ========== CONFIGURATION ========== #
ANGULAR_RESOLUTION = '1.0'  # '0.33', '0.5', or '1.0'
SERIAL_BAUD = 115200        # CDC-ACM baud (has no effect on USB speed but required by driver)
SERIAL_TIMEOUT = 1.0        # seconds
# =================================== #

STX, ETX = b'\x02', b'\x03'

# Defines a named tuple that can hold datagram information from a SICK TiM561-2050101 Lidar device.
Sick561Datagram = collections.namedtuple("sick561_datagram", ["TypeOfCommand", "Command", "VersionNumber",
                                                               "DeviceNumber", "SerialNumber", "DeviceStatus1",
                                                               "DeviceStatus2", "TelegramCounter", "ScanCounter",
                                                               "TimeSinceStartup", "TimeOfTransmission", 
                                                               "InputStatus1", "InputStatus2", "OutputStatus1",
                                                               "OutputStatus2", "ScanningFrequency", 
                                                               "MeasurementFrequency", "NumberOfEncoders",
                                                               "NumberOf16bitChannels", "MeasuredDataContents",
                                                               "ScalingFactor", "ScalingOffset", "StartingAngle",
                                                               "AngularStepWidth", "NumberOfData", "Data"])


def parse_number(nbr_str: bytes) -> int:
    """Parse decimal or hex numbers from bytes."""
    try:
        return int(nbr_str, 16)
    except ValueError:
        return int(nbr_str)


def decode_datagram(datagram: bytes):
    try:
        items = datagram.split(b' ')
        header = {}
        header['TypeOfCommand'] = items[0].decode('ascii')
        header['Command'] = items[1].decode('ascii')
        # Guard: only process scans
        if header['TypeOfCommand'] != 'sSN' or header['Command'] != 'LMDscandata':
            return None
        header['VersionNumber'] = parse_number(items[2])
        header['DeviceNumber'] = parse_number(items[3])
        header['SerialNumber'] = items[4].decode('ascii')
        header['DeviceStatus1'] = parse_number(items[5])
        header['DeviceStatus2'] = parse_number(items[6])
        header['TelegramCounter'] = parse_number(items[7])
        header['TimeSinceStartup'] = parse_number(items[9])
        header['TimeOfTransmission'] = parse_number(items[10])
        header['AngularStepWidth'] = parse_number(items[24])
        header['NumberOfData'] = parse_number(items[25])
        header['Data'] = [parse_number(x) / 1000 for x in items[26:26 + header['NumberOfData']]]
        return header
    except (IndexError, UnicodeDecodeError, ValueError):
        return None


def find_serial_port() -> str | None:
    """Return first likely SICK serial device port, or None if none found."""
    candidates = sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
    return candidates[0] if candidates else None


def sopas_command(cmd_ascii: bytes) -> bytes:
    """Wrap an ASCII SOPAS payload with STX/ETX."""
    return STX + cmd_ascii + ETX


def bytes_from_serial(ser: serial.Serial):
    """Yield bytes from serial port in small chunks."""
    while True:
        data = ser.read(256)
        if not data:
            yield from []
        else:
            for b in data:
                yield bytes([b])


def datagrams_from_serial(ser: serial.Serial):
    """Generate SOPAS datagrams (STX ... ETX) from serial stream."""
    byte_gen = bytes_from_serial(ser)
    while True:
        datagram = b''
        # find STX
        for b in byte_gen:
            if b == STX:
                break
        # read until ETX
        for b in byte_gen:
            if b == ETX:
                break
            datagram += b
        if datagram:
            yield datagram


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


class Lidar():
    def __init__(self, port: str | None = None, angular_resolution: str = ANGULAR_RESOLUTION):
        self.port = port or find_serial_port()
        self.angular_resolution = angular_resolution
        self.stop = False

        self.ds = None  # latest distances
        self.datagrams_generator = None
        self.usb = None  # PyUSB connection when serial is not present
        self._usb_buffer = b''

        # Parameters (match L1_lidar.py semantics)
        self.min_distance = 0.1
        self.max_distance = 3.0
        resolution_map = {'0.33': 811, '0.5': 541, '1.0': 271}
        self.datagram_size = resolution_map.get(angular_resolution, 271)
        self.sensor_angle = 270
        ang_start, ang_stop = -135, 135
        self.angles = np.linspace(ang_start, ang_stop, num=self.datagram_size, endpoint=True)

    def _send_and_wait_ack(self, ser: serial.Serial, cmd_ascii: bytes, expect_prefix: bytes | None = None, tries: int = 10):
        """Send SOPAS command and wait for matching sAN/sEA response (best-effort)."""
        ser.write(sopas_command(cmd_ascii))
        ser.flush()
        # Read several datagrams to catch the answer
        gen = datagrams_from_serial(ser)
        for _ in range(tries):
            try:
                dg = next(gen)
            except StopIteration:
                break
            if expect_prefix is None:
                return dg
            if dg.startswith(expect_prefix):
                return dg
        return b''

    def connect(self):
        """Open serial port if present, else fall back to PyUSB bulk; initialize for streaming."""
        if self.port is not None:
            # Serial transport
            self.ser = serial.Serial(
                port=self.port,
                baudrate=SERIAL_BAUD,
                timeout=SERIAL_TIMEOUT,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                write_timeout=1.0,
            )
            time.sleep(0.2)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            print(f"[+] Connected to LiDAR via USB serial at {self.port}")

            # SOPAS init over serial
            print("[*] Login...")
            self._send_and_wait_ack(self.ser, b'sMN SetAccessMode 03 F4724744', expect_prefix=b'sAN SetAccessMode')
            print("[*] Start measurement...")
            self._send_and_wait_ack(self.ser, b'sMN LMCstartmeas', expect_prefix=b'sAN LMCstartmeas')
            print("[*] Switch to RUN mode...")
            self._send_and_wait_ack(self.ser, b'sMN Run', expect_prefix=b'sAN Run')
            print(f"[*] Configure resolution {self.angular_resolution}°...")
            res_code = {'0.33': '1', '0.5': '2', '1.0': '3'}.get(self.angular_resolution, '3')
            cfg_cmd = f'sMN mLMPsetscancfg +{res_code} +1 -450000 +2250000'.encode()
            self._send_and_wait_ack(self.ser, cfg_cmd, expect_prefix=b'sAN mLMPsetscancfg')
            print("[*] Enable LMDscandata...")
            self._send_and_wait_ack(self.ser, b'sEN LMDscandata 1', expect_prefix=b'sEA LMDscandata')
            self.datagrams_generator = datagrams_from_serial(self.ser)
            print("LiDAR streaming enabled (serial)")
            return

        # Fallback: PyUSB bulk transport
        if not HAVE_USB:
            raise RuntimeError("No serial port found and PyUSB not available. Install pyusb or check /dev/ttyACM*.")

        dev = usb.core.find(idVendor=0x19A2, idProduct=0x5001)
        if dev is None:
            raise RuntimeError("SICK TIM device not found via USB (19a2:5001). Check cabling and lsusb.")
        # Detach kernel drivers on all interfaces if needed
        try:
            for i in range(dev.get_active_configuration().bNumInterfaces):
                if dev.is_kernel_driver_active(i):
                    dev.detach_kernel_driver(i)
        except Exception:
            pass
        dev.set_configuration()
        cfg = dev.get_active_configuration()
        ep_in, ep_out = None, None
        # Search all interfaces for a usable IN and OUT endpoint (prefer BULK)
        for intf in cfg:
            for ep in intf:
                # 0x02 = bulk, 0x03 = interrupt; accept either if necessary
                direction = usb.util.endpoint_direction(ep.bEndpointAddress)
                if direction == usb.util.ENDPOINT_IN and ep_in is None:
                    ep_in = ep
                if direction == usb.util.ENDPOINT_OUT and ep_out is None:
                    ep_out = ep
            if ep_in is not None and ep_out is not None:
                break
        if ep_in is None or ep_out is None:
            raise RuntimeError("USB endpoints not found for TIM device (no IN/OUT endpoints)")

        self.usb = {
            'dev': dev,
            'ep_in': ep_in,
            'ep_out': ep_out,
        }
        print(f"[+] Connected to LiDAR via USB bulk (IN=0x{ep_in.bEndpointAddress:02x}, OUT=0x{ep_out.bEndpointAddress:02x})")

        def usb_write(payload_ascii: bytes):
            frame = sopas_command(payload_ascii)
            ep_out.write(frame)
            time.sleep(0.01)

        def usb_expect(prefix: bytes, tries: int = 20) -> bytes:
            start = time.time()
            while time.time() - start < 2.0:
                try:
                    chunk = bytes(ep_in.read(512, timeout=500))
                except Exception as e:
                    # Gracefully ignore USB timeout errors to allow progressing
                    # even if a specific ACK is not returned by this device.
                    time.sleep(0.01)
                    continue
                if chunk:
                    self._usb_buffer += chunk
                    while STX in self._usb_buffer and ETX in self._usb_buffer:
                        s = self._usb_buffer.find(STX)
                        e = self._usb_buffer.find(ETX, s)
                        if e > s:
                            dg = self._usb_buffer[s+1:e]
                            self._usb_buffer = self._usb_buffer[e+1:]
                            if prefix is None or dg.startswith(prefix):
                                return dg
                        else:
                            break
            return b''

        # SOPAS init over USB bulk
        print("[*] Login...")
        usb_write(b'sMN SetAccessMode 03 F4724744')
        ack = usb_expect(b'sAN SetAccessMode')
        print("    ack:", ack[:40] if ack else b'')
        print("[*] Start measurement...")
        usb_write(b'sMN LMCstartmeas')
        ack = usb_expect(b'sAN LMCstartmeas')
        print("    ack:", ack[:40] if ack else b'')
        print("[*] Switch to RUN mode...")
        usb_write(b'sMN Run')
        ack = usb_expect(b'sAN Run')
        print("    ack:", ack[:40] if ack else b'')
        print(f"[*] Configure resolution {self.angular_resolution}°...")
        res_code = {'0.33': '1', '0.5': '2', '1.0': '3'}.get(self.angular_resolution, '3')
        usb_write(f'sMN mLMPsetscancfg +{res_code} +1 -450000 +2250000'.encode())
        ack = usb_expect(b'sAN mLMPsetscancfg')
        print("    ack:", ack[:60] if ack else b'')
        if not ack:
            print("    (no mLMPsetscancfg ACK — continuing; this model may not support it)")
        print("[*] Enable LMDscandata...")
        usb_write(b'sEN LMDscandata 1')
        ack = usb_expect(b'sEA LMDscandata')
        print("    ack:", ack[:40] if ack else b'')
        print("LiDAR streaming enabled (USB bulk)")

        # For USB bulk, the run() loop will read from ep_in and parse frames.
        self.datagrams_generator = None  # not used for USB bulk

    def get(self, num_points=108):
        if self.ds is None:
            return None
        # Ensure distances and angles arrays are aligned in length
        n = min(len(self.ds), len(self.angles))
        ds = self.ds[:n]
        angs = self.angles[:n]
        data = np.zeros((num_points, 2))
        partitioned_distances = np.array_split(ds, num_points)
        partitioned_angles = np.array_split(angs, num_points)
        for idx in range(num_points):
            min_idx = np.argmin(partitioned_distances[idx])
            data[idx][0] = partitioned_distances[idx][min_idx]
            data[idx][1] = partitioned_angles[idx][min_idx]
        return data

    def kill(self, processor):
        self.stop = True
        processor.join()
        try:
            # Disable stream politely
            self.ser.write(sopas_command(b'sEN LMDscandata 0'))
            self.ser.flush()
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass
        print("LiDAR disconnected")

    def clean_datagram_by_distance(self, ds):
        ds = np.where(ds > self.min_distance, ds, -1)
        ds = np.where(ds < self.max_distance, ds, -1)
        return ds

    def clean_datagram_by_angle(self, ds, viewAngle=30):
        cleanArraySize = int((self.datagram_size/self.sensor_angle)*(viewAngle/2))
        midpoint = len(ds) // 2
        start = midpoint - (cleanArraySize // 2)
        if cleanArraySize % 2 == 0:
            start -= 1
        end = start + cleanArraySize
        return ds[start:end]

    @threaded
    def run(self):
        print("LiDAR started")
        if self.usb is None:
            # Serial path
            while not self.stop:
                try:
                    datagram = next(self.datagrams_generator)
                    decoded = decode_datagram(datagram)
                    if decoded is not None:
                        self.ds = np.array(decoded['Data'])
                except Exception:
                    time.sleep(0.01)
        else:
            # USB bulk path
            ep_in = self.usb['ep_in']
            while not self.stop:
                try:
                    chunk = bytes(ep_in.read(8192, timeout=1000))
                    if not chunk:
                        continue
                    self._usb_buffer += chunk
                    while STX in self._usb_buffer and ETX in self._usb_buffer:
                        s = self._usb_buffer.find(STX)
                        e = self._usb_buffer.find(ETX, s)
                        if e > s:
                            dg = self._usb_buffer[s+1:e]
                            self._usb_buffer = self._usb_buffer[e+1:]
                            decoded = decode_datagram(dg)
                            if decoded is not None:
                                self.ds = np.array(decoded['Data'])
                        else:
                            break
                except Exception:
                    time.sleep(0.01)
        print("LiDAR stopped")
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read SICK TiM LiDAR over USB and print data")
    parser.add_argument("--print", dest="print_mode", choices=["closest", "raw", "pairs", "pairs-all"], default="closest",
                        help="What to print each cycle: closest point, raw distances array, or (angle, distance) pairs")
    parser.add_argument("--num-points", type=int, default=108, help="Number of downsampled points for pairs mode")
    parser.add_argument("--count", type=int, default=0, help="Number of print cycles before exiting (0=infinite)")
    args = parser.parse_args()

    lidarsensor = Lidar()
    lidarsensor.connect()
    processor = lidarsensor.run()
    time.sleep(1)
    printed = 0
    try:
        def do_print():
            if args.print_mode == "closest":
                data = lidarsensor.get(args.num_points)
                if data is None:
                    print("Waiting for data...")
                    return False
                min_idx = int(np.argmin(data[:, 0]))
                print(f"Got {len(data)} points | Closest: {data[min_idx, 0]:.2f}m @ {data[min_idx, 1]:.1f}°")
                return True
            elif args.print_mode == "raw":
                ds = lidarsensor.ds
                if ds is None:
                    print("Waiting for data...")
                    return False
                print(f"Raw distances ({len(ds)}):", np.round(ds, 3))
                return True
            elif args.print_mode == "pairs":
                data = lidarsensor.get(args.num_points)
                if data is None:
                    print("Waiting for data...")
                    return False
                pairs = [(float(a), float(d)) for d, a in data]
                print(f"Pairs (angle°, distance m) count={len(pairs)}:\n", pairs)
                return True
            else:  # pairs-all
                ds = lidarsensor.ds
                if ds is None:
                    print("Waiting for data...")
                    return False
                # Align and emit full-resolution angle-distance pairs
                n = min(len(ds), len(lidarsensor.angles))
                angs = lidarsensor.angles[:n]
                ds = ds[:n]
                pairs = [(float(angs[i]), float(ds[i])) for i in range(n)]
                print(f"Pairs (angle°, distance m) count={len(pairs)}:\n", pairs)
                return True

        if args.count > 0:
            for _ in range(args.count):
                time.sleep(0.5)
                did = do_print()
                if not did:
                    continue
        else:
            while True:
                time.sleep(0.5)
                do_print()
    except KeyboardInterrupt:
        print("\nStopping LiDAR...")
    finally:
        lidarsensor.kill(processor)
