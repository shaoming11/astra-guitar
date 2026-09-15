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
# Sent at a single baud to tell a self-echo apart from a real reply: if what comes
# back tracks what goes out, nothing but the adapter is driving the line.
ECHO_PROBES = [bytes([0x55] * 6), bytes([0xAA] * 6), bytes([0xFF] * 6), bytes([0x00] * 6)]

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

# Echo test: vary the payload at one baud and see whether the reply follows it.
echo_hits = 0
try:
    s = serial.Serial(PORT, 115200, timeout=0.25)
    print("\necho test @115200 (does the reply track the payload?)")
    for p in ECHO_PROBES:
        time.sleep(0.03); s.reset_input_buffer()
        s.write(p); s.flush(); time.sleep(0.12)
        r = s.read(64)
        same = r == p
        inv = r == bytes(b ^ 0xFF for b in p)
        if same or inv: echo_hits += 1
        print(f"  {p.hex():<14} -> {r.hex() or '(nothing)':<14}"
              f"{'  EXACT ECHO' if same else '  INVERTED ECHO' if inv else ''}")
    s.close()
except Exception as e:
    print("echo test failed:", e)

print()
if any(bytes.fromhex(v)[:2] == HDR for v in distinct):
    print("VERDICT: real Feetech framing found - the bus works; it is a baud/ID/model mismatch.")
elif not distinct:
    print("VERDICT: line idle. Check servo power and the board->first-servo cable.")
elif echo_hits >= 2:
    print("VERDICT: the adapter is hearing its own transmission - nothing else drives the")
    print("         line, so no servo is replying. This does NOT single out the board.")
    print("         Three candidates, roughly equally likely:")
    print("           1. adapter's half-duplex direction control stuck (board fault)")
    print("           2. servo logic unpowered (LEDs can light off a rail too weak to run")
    print("              the servo MCU) - measure VCC-GND at the first servo connector")
    print("           3. broken data conductor board -> first servo (power wires are separate)")
    print("         Cheapest disambiguation: try a known-good arm on this board, or a")
    print("         spare board on this arm.")
else:
    print("VERDICT: bytes come back but never frame correctly and do not track the payload.")
    print("         Marginal signal integrity - suspect the data line or the transceiver.")
