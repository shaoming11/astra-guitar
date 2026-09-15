"""Brute-force scan: every supported model/protocol x every baud rate."""
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech.tables import MODEL_BAUDRATE_TABLE, MODEL_PROTOCOL
PORT = "/dev/cu.usbmodem5A7C1220421"
hits = []
for model in ["sts3215", "sts3250", "sm8512bl", "scs0009"]:
    bus = FeetechMotorsBus(port=PORT, motors={"m": Motor(1, model, MotorNormMode.RANGE_M100_100)},
                           protocol_version=MODEL_PROTOCOL[model])
    bus._connect(handshake=False)
    for br in sorted(MODEL_BAUDRATE_TABLE[model], reverse=True):
        try:
            bus.set_baudrate(br)
            found = sorted(bus.broadcast_ping() or {})
        except Exception as e:
            found = []
        flag = "  <<< FOUND" if found else ""
        print(f"{model:>9} @ {br:>8}: {found}{flag}", flush=True)
        if found: hits.append((model, br, found))
    bus.disconnect(disable_torque=False)
print("\nRESULT:", hits if hits else "nothing responded on any model/baud combination")
