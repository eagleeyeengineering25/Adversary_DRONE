# LiDAR USB & SSH Tunnel Setup Guide

## 🔧 What Changed

### 1. **L1_lidar_GUI.py** - Local GUI (Ethernet or USB)
- ✅ Uses `L1_lidar_usb.py` under the hood for USB when run with `--usb`
- ✅ Proper login sequence (`SetAccessMode`) and framing handled centrally
- ✅ Buffer-based datagram parsing and robust timeouts via shared code

### 2. **L1_lidar_GUI_tunnel.py** - NEW! SSH Tunnel + Obstacle Avoidance
- ✅ Based on `ssh_feed.py` architecture
- ✅ Server mode: Reads LiDAR via USB on Jetson
- ✅ Client mode: Displays GUI on laptop via SSH tunnel
- ✅ **Obstacle avoidance warnings:**
  - 🔴 RED: < 0.5m (DANGER)
  - 🟠 ORANGE: < 1.0m (WARNING)
  - 🟡 YELLOW: < 2.0m (CAUTION)
  - 🟢 GREEN: > 2.0m (Clear)
- ✅ Serializes numpy arrays with pickle for network transfer
- ✅ Color-coded point cloud visualization

---

## 📋 Files Summary

| File | Purpose | Connection |
|------|---------|------------|
| `L1_lidar_usb.py` | USB reader + print modes (closest/raw/pairs/pairs-all) | USB only |
| `L1_lidar_GUI.py` | Local GUI (Ethernet or USB via --usb) | Ethernet/USB |
| `L1_lidar_GUI_tunnel.py` | Server+Client (Jetson server + optional client) | USB → SSH tunnel |
| `L1_lidar_GUI_client.py` | Client-only (run on laptop) | SSH tunnel |

---

## 🚀 Quick Start

### Option 1: Local USB GUI
```bash
# On Jetson (connected to LiDAR via USB)
cd ~/Adversary_DRONE/Lidar_Lab
sudo python3 L1_lidar_GUI.py --usb
```

Tip: Use `--resolution 1.0` (270 points, 50 Hz) for fastest GUI.

---

### Option 2: Remote GUI via SSH Tunnel (NEW!)

#### **Step 1: On Jetson (Server)**
```bash
cd ~/Adversary_DRONE/Lidar_Lab
sudo python3 L1_lidar_GUI_tunnel.py --server --port 5100
```

**You should see:**
```
[server] listening on ('0.0.0.0', 5100)
[+] Connected to USB device 0x19a2:0x5001
[*] Attempting login...
[*] Setting angular resolution to 1.0°...
[*] Activating data streaming...
[*] Streaming LiDAR data to client...
```

#### **Step 2: On Laptop (Client)**

**A) Linux/Mac:**
```bash
# Terminal 1: SSH tunnel (keep running)
ssh -L 5100:localhost:5100 eagle@10.250.240.81

# Terminal 2: Run client
cd ~/Downloads  # or wherever you copied the file
    python3 L1_lidar_GUI_tunnel.py --client --host localhost --port 5100
```

**B) Windows PowerShell:**
```powershell
# One-liner (starts tunnel in background + GUI)
Start-Job { ssh -N -L 5100:127.0.0.1:5100 eagle@10.250.240.81 } ; cd "$env:USERPROFILE\Downloads" ; python L1_lidar_GUI_tunnel.py --client --host 127.0.0.1 --port 5100
```

---

## 🔑 SSH Key Setup (One-Time)

**On Windows (PowerShell):**
```powershell
# 1. Generate key
ssh-keygen -t ed25519

# 2. Copy to Jetson
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh eagle@10.250.240.81 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

# 3. Test (should not ask for password)
ssh eagle@10.250.240.81
```

**On Linux/Mac:**
```bash
# 1. Generate key
ssh-keygen -t ed25519

# 2. Copy to Jetson
ssh-copy-id eagle@10.250.240.81

# 3. Test
ssh eagle@10.250.240.81
```

---

## 🐛 Troubleshooting

### USB Connection Issues
### Copy the client script to your laptop

On your laptop (Linux/Mac):
```bash
scp eagle@<jetson-ip>:/home/eagle/Adversary_DRONE/Lidar_Lab/L1_lidar_GUI_client.py ~/Downloads/
```

On Windows PowerShell (with OpenSSH):
```powershell
scp eagle@10.250.240.81:/home/eagle/Adversary_DRONE/Lidar_Lab/L1_lidar_GUI_client.py C:\Users\savya\Downloads\
```

Then run the client from Downloads after starting the server on the Jetson:
```bash
python3 ~/Downloads/L1_lidar_GUI_client.py --host localhost --port 5100
```

**Problem:** `USB device 0x19a2:0x5001 not found`

**Solution:**
```bash
# 1. Check if LiDAR is connected
lsusb | grep -i sick

# Should show:
# Bus 001 Device 005: ID 19a2:5001 SICK AG

# 2. Check permissions
ls -l /dev/bus/usb/001/005  # (use your bus/device numbers)

# 3. Run with sudo (USB only)
sudo python3 L1_lidar_GUI.py --usb
```

---

### SSH Tunnel Issues

**Problem:** Client can't connect to localhost:5100

**Solution:**
```bash
# 1. Check if tunnel is running
netstat -an | grep 5100  # Linux/Mac
netstat -an | findstr 5100  # Windows

# 2. Check if server is running on Jetson
ssh eagle@10.250.240.81
netstat -an | grep 5100
# Should show: tcp 0 0 0.0.0.0:5100 0.0.0.0:* LISTEN

# 3. Restart tunnel
pkill ssh  # Kill old tunnels
ssh -L 5100:localhost:5100 eagle@10.250.240.81
```

---

### No Data / Frozen GUI

**Problem:** GUI opens but shows "Waiting for data..."

**Solution:**
```bash
# 1. Check server logs on Jetson
# Should see: [server] client connected: ('127.0.0.1', xxxxx)

# 2. Check USB buffer overflow
# On server, look for "[server] error: ..." messages

# 3. Increase timeout (edit file):
data = usb_conn.read(size=8192, timeout=5000)  # Change from 1000 to 5000
```

---

## 📊 Key Protocol Differences

### Working USB Protocol (see L1_lidar_usb.py)

```python
# 1. Login command
usb_conn.write(b'\x02sMN SetAccessMode 03 F4724744\x03')
time.sleep(0.3)

# 2. Configure angular resolution
cmd = f'\x02sMN mLMPsetscancfg +{freq_code} +1 -450000 +2250000\x03'.encode()
usb_conn.write(cmd)
time.sleep(0.3)

# 3. Start streaming
usb_conn.write(b'\x02sEN LMDscandata 1\x03')
time.sleep(0.3)

# 4. Read with large buffer
data = usb_conn.read(size=8192, timeout=2000)
```

**Key points:**
- ✅ No trailing `\0` byte (unlike Ethernet)
- ✅ Login step is REQUIRED for USB
- ✅ Wait 300ms between commands
- ✅ Use 8KB buffer (not 256 bytes)
- ✅ 2 second timeout for stability

---

## 🎯 Obstacle Avoidance Features (tunnel.py)

### Distance Thresholds
```python
if min_dist < 0.5:
    # 🔴 RED - STOP IMMEDIATELY
    status = "⚠️ DANGER! Obstacle at {min_dist:.2f}m"

elif min_dist < 1.0:
    # 🟠 ORANGE - SLOW DOWN
    status = "⚠️ WARNING: Close obstacle"

elif min_dist < 2.0:
    # 🟡 YELLOW - BE AWARE
    status = "⚠️ CAUTION: Obstacle nearby"

else:
    # 🟢 GREEN - SAFE TO PROCEED
    status = "✓ Clear"
```

### Point Color Coding
- **Red points**: < 1.0m (immediate danger)
- **Orange points**: 1.0-2.0m (caution zone)
- **Cyan points**: > 2.0m (safe)

---

## 📁 File Copy Commands

### Copy tunnel.py to laptop (one-time):
```bash
# Linux/Mac
scp eagle@10.250.240.81:/home/eagle/Adversary_DRONE/Lidar_Lab/L1_lidar_GUI_tunnel.py ~/Downloads/

# Windows PowerShell
scp eagle@10.250.240.81:/home/eagle/Adversary_DRONE/Lidar_Lab/L1_lidar_GUI_tunnel.py C:\Users\savya\Downloads\
```

### Install dependencies on laptop:
```bash
pip3 install numpy matplotlib pyusb
```

---

## ✅ Testing Checklist

- [ ] USB connection works (`lsusb` shows SICK device)
- [ ] `L1_lidar_usb.py` prints scan data (e.g., `--print pairs`)
- [ ] `L1_lidar_GUI.py --usb` shows live visualization
- [ ] SSH key authentication works (no password prompt)
- [ ] SSH tunnel established (port 5100 listening)
- [ ] Remote GUI connects and displays scans
- [ ] Obstacle warnings appear when moving hand near LiDAR

---

## 🔧 Advanced: Change Angular Resolution

Edit the configuration at the top of any file:

```python
# Options: '0.33', '0.5', '1.0'
ANGULAR_RESOLUTION = '1.0'
```

| Resolution | Points | Scan Rate | Use Case |
|------------|--------|-----------|----------|
| 0.33° | 810 | 15 Hz | High detail mapping |
| 0.5° | 540 | 25 Hz | Balanced |
| 1.0° | 270 | 50 Hz | Fast obstacle avoidance |

**Recommendation:** Use `1.0°` for real-time obstacle avoidance (fastest).

---

## 📞 Need Help?

1. Check server terminal for error messages
2. Check client terminal for connection issues
3. Verify USB device: `lsusb | grep SICK`
4. Verify SSH tunnel: `netstat -an | grep 5100`
5. Test with `L1_lidar_usb.py --print pairs` first (simplest case)

---

## 🎉 Success Indicators

**Server (Jetson):**
```
[+] Connected to USB device 0x19a2:0x5001
[*] Activating data streaming...
[server] client connected: ('127.0.0.1', 54321)
[*] Streaming LiDAR data to client...
```

**Client (Laptop):**
```
[client] connecting to ('localhost', 5100)...
[client] connected - starting GUI
[GUI shows live polar plot with colored points]
Status: ✓ Clear - Nearest: 3.45m
```

**Move your hand near the LiDAR:**
- Points turn RED
- Status shows "⚠️ DANGER! Obstacle at 0.35m"

---

That's it! You now have:
- ✅ Working USB LiDAR connection
- ✅ Remote GUI via SSH tunnel
- ✅ Real-time obstacle avoidance warnings

🚀 Ready for autonomous navigation!
