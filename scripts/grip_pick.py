"""Open the gripper for 2 s, close on the pick, and hold forever.

Drives ONLY motor 6 (gripper) over the raw Feetech bus - no arm calibration needed.
Torque is limited so the servo stalls gently on the pick instead of stripping.
Ctrl+C exits the script but leaves torque ON so the pick stays clamped.
"""
import sys, time
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

PORT = "/dev/cu.usbmodem5A7C1220421"
OPEN_TICKS = 3400        # from my_arm.json gripper range (2029..3557); open ~ range_max
CLOSE_TICKS = 2029       # commanded past the pick -> servo stalls on it (that's the clamp)
TORQUE_LIMIT = 350       # of 1000. ~35% - firm on a pick, safe for the STS3215
OPEN_SECONDS = 2.0

bus = FeetechMotorsBus(port=PORT, motors={"gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)})
bus.connect(handshake=False)
g = "gripper"

bus.write("Torque_Enable", g, 0)
bus.write("Acceleration", g, 20)
bus.write("Max_Torque_Limit", g, TORQUE_LIMIT)
bus.write("Torque_Limit", g, TORQUE_LIMIT)
bus.write("Torque_Enable", g, 1)

print(f"start pos: {bus.read('Present_Position', g, normalize=False)}")
print(f"OPEN -> {OPEN_TICKS}  (put the pick in now)")
bus.write("Goal_Position", g, OPEN_TICKS, normalize=False)
time.sleep(OPEN_SECONDS)

print(f"CLOSE -> {CLOSE_TICKS}  (holding)")
bus.write("Goal_Position", g, CLOSE_TICKS, normalize=False)
time.sleep(1.0)
print(f"settled at: {bus.read('Present_Position', g, normalize=False)}  - stalled on pick if not {CLOSE_TICKS}")

try:
    while True:
        time.sleep(5)
        bus.write("Torque_Enable", g, 1)   # re-assert in case of a protective trip
        bus.write("Goal_Position", g, CLOSE_TICKS, normalize=False)
except KeyboardInterrupt:
    pass
finally:
    bus.disconnect(disable_torque=False)   # keep clamping after exit
    print("exited; torque left ON, pick still held")
