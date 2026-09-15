"""Poll the servo bus until motors appear. Wiggle cables and watch this print."""
import time, os, sys
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode
PORT = "/dev/cu.usbmodem5A7C1220421"
last = None
for i in range(150):
    if not os.path.exists(PORT):
        state = "no USB port"
    else:
        try:
            bus = FeetechMotorsBus(port=PORT, motors={"m": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)})
            bus._connect(handshake=False)
            found = sorted(bus.broadcast_ping() or {})
            bus.disconnect(disable_torque=False)
            state = f"MOTORS: {found}" if found else "port ok, bus silent"
        except Exception as e:
            state = f"error: {type(e).__name__}"
    if state != last:
        print(f"[{time.strftime('%H:%M:%S')}] {state}", flush=True)
        last = state
    time.sleep(2)
