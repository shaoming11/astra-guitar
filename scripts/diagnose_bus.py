"""Decide whether a silent servo bus is a dead adapter board or a wiring problem.

Sweeps raw serial over every plausible baud rate and looks for valid Feetech framing
(a reply starts FF FF). The key tell is NOT whether bytes come back - a dead board still
returns garbage - but whether those bytes CHANGE with the baud rate:

  bytes change across bauds .... real data is on the line; it is a baud/ID/model mismatch
  bytes identical across bauds . no data at all; the UART is framing a stuck DC level,
                                 i.e. the adapter's transceiver is dead -> replace the board
  nothing at all ............... line held at idle; check servo power and the first cable
"""
import serial, sys, time

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbmodem5A7C1220421"
BAUDS = [9600, 19200, 38400, 57600, 76800, 115200, 128000, 153600, 230400, 250000,
         256000, 460800, 500000, 512000, 750000, 921600, 1000000, 1500000, 2000000]
PING = bytes([0xFF, 0xFF, 0xFE, 0x02, 0x01, 0xFE])   # broadcast ping
HDR = b"\xff\xff"

seen = {}
for b in BAUDS:
    try:
        s = serial.Serial(PORT, b, timeout=0.2)
    except Exception as e:
        print(f"{b:>8}: cannot open ({e})"); continue
    time.sleep(0.03); s.reset_input_buffer()
    s.write(PING); s.flush(); time.sleep(0.12)
    rx = s.read(64); s.close()
    seen[b] = rx.hex()
    tag = "  <<< VALID HEADER" if HDR in rx else ""
    print(f"{b:>8}: {rx.hex() or '(nothing)':<28}{tag}")

distinct = set(seen.values()) - {""}
print()
if any(bytes.fromhex(v)[:2] == HDR for v in distinct):
    print("VERDICT: real Feetech framing found - the bus works; it is a baud/ID/model mismatch.")
elif not distinct:
    print("VERDICT: line idle. Check servo power and the board->first-servo cable.")
elif len(distinct) <= 2:
    print("VERDICT: identical bytes across a wide baud range = no real data on the line.")
    print("         The adapter's transceiver is dead. Replace the board")
    print("         (Waveshare Bus Servo Adapter (A), or Feetech FE-URT-1).")
else:
    print("VERDICT: bytes vary but never frame correctly - marginal signal integrity.")
    print("         Suspect the data line or a failing transceiver.")
