from dronekit import connect, VehicleMode
from pymavlink import mavutil
import time
import math

# --- Helper function to convert Euler angles to quaternion ---
def to_quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    """
    Convert Euler angles (radians) to quaternion [w, x, y, z]
    """
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

# --- Connect to Cube Orange ---
print("Connecting to Cube Orange...")
vehicle = connect('/dev/ttyACM0', baud=57600, wait_ready=True)
print("Connected:", vehicle.version)

# --- Set mode to GUIDED_NOGPS ---
print("Setting mode to GUIDED_NOGPS...")
vehicle.mode = VehicleMode("GUIDED_NOGPS")
while not vehicle.mode.name == "GUIDED_NOGPS":
    print("Waiting for mode change...")
    time.sleep(1)
print("Mode set to GUIDED_NOGPS")

# --- Arm motors (NO PROPS!) ---
vehicle.armed = True
while not vehicle.armed:
    print("Waiting for arming...")
    time.sleep(1)
print("Vehicle armed!")

# # --- Safety delay before starting commands ---
# print("\n*** Waiting 10 seconds before sending commands (safety delay) ***")
# for i in range(10, 0, -1):
#     print(f"Starting in {i}...")
#     time.sleep(1)
# print("Starting test sequence now!\n")

# # --- Function to send body-frame velocity ---
# def send_velocity(vx, vy, vz, duration=1):
#     """
#     vx: forward
#     vy: right
#     vz: down
#     duration: seconds
#     """
#     msg = vehicle.message_factory.set_position_target_local_ned_encode(
#         0, 0, 0,
#         mavutil.mavlink.MAV_FRAME_BODY_NED,
#         0b0000111111000111,  # ignore everything except velocity
#         0, 0, 0,
#         vx, vy, vz,
#         0, 0, 0,
#         0, 0
#     )
#     for _ in range(int(duration * 10)):  # 10 Hz
#         vehicle.send_mavlink(msg)
#         vehicle.flush()
#         print(f"Sent velocity: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")
#         print("Measured velocity:", vehicle.velocity)
#         time.sleep(0.1)

# --- Function to send attitude/thrust commands ---
def send_attitude_target(roll=0.0, pitch=0.0, yaw_rate=0.0, thrust=0.0, duration=1):
    """
    roll, pitch in radians
    yaw_rate in rad/s
    thrust: 0.0-1.0
    duration: seconds
    """
    q = to_quaternion(roll, pitch, 0)

    print(f"Quaternion (q): w={q[0]:.4f}, x={q[1]:.4f}, y={q[2]:.4f}, z={q[3]:.4f}")

    msg = vehicle.message_factory.set_attitude_target_encode(
        0,
        0, 0,
        0b00000111,  # ignore body rates except yaw_rate
        q,
        0, 0, yaw_rate,
        thrust
    )
    for _ in range(int(duration * 10)):
        vehicle.send_mavlink(msg)
        vehicle.flush()
        print(f"Sent attitude: roll={roll}, pitch={pitch}, yaw_rate={yaw_rate}, thrust={thrust}")
        time.sleep(0.1)

# --- Test sequence ---

# # 1. Test forward/back/left/right with velocity
# print("\n--- Testing forward motion ---")
# send_velocity(0.3, 0, 0, duration=3)

# print("\n--- Testing backward motion ---")
# send_velocity(-0.3, 0, 0, duration=3)

# print("\n--- Testing right motion ---")
# send_velocity(0, 0.3, 0, duration=3)

# print("\n--- Testing left motion ---")
# send_velocity(0, -0.3, 0, duration=3)

# 2. Hover/throttle test
print("\n--- Hover throttle test ---")
send_attitude_target(thrust=0.2, duration=5)  # mid throttle (hover-level)

send_attitude_target(thrust=0, duration=5) 

print("\n--- Pitch test ---")
send_attitude_target(pitch=0.2, duration=5) 

send_attitude_target(pitch=-0.2, duration=5) 

print("\n--- Yaw rate test ---")
send_attitude_target(yaw_rate=0.2, duration=5) 

# send_attitude_target(yaw_rate=-0.3, duration=5)  

# # 3. Stop motion
# print("\n--- Stopping (hover/idle) ---")
# send_velocity(0, 0, 0, duration=3)

# --- Switch to STABILIZE mode ---
print("\nSwitching to STABILIZE mode...")
vehicle.mode = VehicleMode("STABILIZE")
while not vehicle.mode.name == "STABILIZE":
    print("Waiting for STABILIZE mode...")
    time.sleep(1)
print("Mode set to STABILIZE.")

# --- Disarm ---
print("Disarming...")
vehicle.armed = False
while vehicle.armed:
    time.sleep(1)
print("Disarmed.")

vehicle.close()
print("Test complete.")
