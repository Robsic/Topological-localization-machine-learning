#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ublox_set_10hz_profile.py
#
# Configure u-blox receivers (e.g., VK-162) for 10 Hz with an optimized NMEA profile:
#   - Navigation rate: 10 Hz (UBX-CFG-RATE)
#   - NMEA on USB:
#       GGA = 1  (=> 10 Hz)
#       VTG = 1  (=> 10 Hz)
#       RMC = 10 (=> 1 Hz)
#       GSA = 0  (off)
#       GSV = 0  (off)
#   - Persist configuration (UBX-CFG-CFG), when supported
#   - Verify output rate via GGA over a configurable time window
#
# Note about UBX-CFG-MSG ("new" format):
#   rateUSB is "every N navigation cycles".
#     N = 1  -> every cycle (if nav=10 Hz => 10/s)
#     N = 10 -> every 10 cycles (if nav=10 Hz => 1/s)
#
# Requirements: pip install pyserial

import argparse
import statistics
import struct
import sys
import time
from pathlib import Path

try:
    import serial
except ImportError:
    print("Missing dependency: pyserial. Install with: pip install pyserial", file=sys.stderr)
    sys.exit(1)

# ------------------------ UBX helpers ------------------------

def ubx_checksum(payload: bytes) -> bytes:
    ck_a = 0
    ck_b = 0
    for b in payload:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return bytes([ck_a, ck_b])


def build_ubx(msg_class: int, msg_id: int, payload: bytes) -> bytes:
    length = struct.pack("<H", len(payload))
    header = bytes([0xB5, 0x62, msg_class, msg_id]) + length
    checksum = ubx_checksum(bytes([msg_class, msg_id]) + length + payload)
    return header + payload + checksum


def send_and_wait_ack(
    ser: serial.Serial,
    msg_class: int,
    msg_id: int,
    payload: bytes,
    timeout_s: float = 1.0
) -> bool:
    """
    Send a UBX packet and wait for ACK-ACK (0x05 0x01).
    Returns True if an ACK is received for the same class/ID.
    """
    frame = build_ubx(msg_class, msg_id, payload)
    ser.write(frame)
    ser.flush()

    t0 = time.time()
    buffer = b""

    while (time.time() - t0) < timeout_s:
        data = ser.read(256)
        if not data:
            continue

        buffer += data

        # ACK-ACK: B5 62 05 01 02 00 <cls> <id> <ckA> <ckB>
        idx = buffer.find(b"\xB5\x62\x05\x01\x02\x00")
        if idx != -1 and len(buffer) >= (idx + 10):
            ack_class = buffer[idx + 6]
            ack_id = buffer[idx + 7]
            return (ack_class == msg_class and ack_id == msg_id)

    return False

# ------------------------ Config functions ------------------------

def cfg_rate(ser: serial.Serial, nav_hz: int) -> bool:
    """
    UBX-CFG-RATE (0x06 0x08)
      measRate = 1000/nav_hz (ms)
      navRate  = 1
      timeRef  = 1 (GPS time)
    """
    if nav_hz <= 0 or nav_hz > 20:
        raise ValueError("Invalid nav_hz (expected 1..20)")

    meas_ms = int(round(1000 / nav_hz))
    payload = struct.pack("<HHH", meas_ms, 1, 1)

    ok = send_and_wait_ack(ser, 0x06, 0x08, payload, timeout_s=1.2)
    print(f"CFG-RATE {nav_hz} Hz:", "OK" if ok else "NO ACK")
    return ok


def cfg_msg_usb_rate_new(ser: serial.Serial, msg_class: int, msg_id: int, usb_rate: int) -> bool:
    """
    UBX-CFG-MSG (0x06 0x01) new format (8 bytes):
      [cls, id, rateI2C, rateUART1, rateUART2, rateUSB, rateSPI, reserved]
    usb_rate:
      1  -> every navigation cycle
      10 -> every 10 navigation cycles
      0  -> disable
    """
    payload = bytes([msg_class, msg_id, 0, 0, 0, usb_rate, 0, 0])
    return send_and_wait_ack(ser, 0x06, 0x01, payload, timeout_s=0.8)


def cfg_msg_legacy(ser: serial.Serial, msg_class: int, msg_id: int, rate: int) -> bool:
    """
    UBX-CFG-MSG legacy (3 bytes): [cls, id, rate]
    """
    payload = bytes([msg_class, msg_id, rate])
    return send_and_wait_ack(ser, 0x06, 0x01, payload, timeout_s=0.8)


def set_nmea_usb_profile(
    ser: serial.Serial,
    gga_usb: int = 1,
    vtg_usb: int = 1,
    rmc_usb: int = 10,
    gsa_usb: int = 0,
    gsv_usb: int = 0
) -> bool:
    """
    Apply NMEA rates on the USB interface:
      gga_usb=1  -> GGA every cycle (10 Hz if nav=10 Hz)
      vtg_usb=1  -> VTG every cycle
      rmc_usb=10 -> RMC every 10 cycles (~1 Hz if nav=10 Hz)
      gsa/gsv=0  -> disabled

    Tries the new CFG-MSG format; falls back to legacy if needed.
    """
    plan = [
        (0xF0, 0x00, "GGA", gga_usb),
        (0xF0, 0x05, "VTG", vtg_usb),
        (0xF0, 0x04, "RMC", rmc_usb),
        (0xF0, 0x02, "GSA", gsa_usb),
        (0xF0, 0x03, "GSV", gsv_usb),
    ]

    all_ok = True
    for cls, mid, name, rate in plan:
        ok = cfg_msg_usb_rate_new(ser, cls, mid, rate)
        tag = "new"
        if not ok:
            ok = cfg_msg_legacy(ser, cls, mid, rate)
            tag = "legacy" if ok else "failed"
        print(f"CFG-MSG {name} -> {rate} ({tag})")
        all_ok &= ok

    return all_ok


def save_cfg(ser: serial.Serial) -> bool:
    """
    Persist CFG_MSG (bit1) and CFG_RATE (bit8) via UBX-CFG-CFG (0x06 0x09).
    Note: some USB dongles won't persist; if settings reset after unplug/replug,
    rerun this script.
    """
    clear_mask = 0x00000000
    save_mask = (1 << 1) | (1 << 8)  # CFG_MSG + CFG_RATE
    load_mask = 0x00000000

    payload = struct.pack("<III", clear_mask, save_mask, load_mask)
    ok = send_and_wait_ack(ser, 0x06, 0x09, payload, timeout_s=1.5)
    print("CFG-CFG (save):", "OK" if ok else "NO ACK (may not persist over USB)")
    return ok

# ------------------------ Verification ------------------------

def verify_rate(
    ser: serial.Serial,
    seconds: float = 6.0,
    log_path: Path = Path("verify_nmea.log")
):
    """
    Count GGA/VTG/RMC for ~seconds and estimate Hz using GGA (one per navigation cycle).
    """
    print(f"\nVerifying output rate for ~{seconds:.0f}s (via GGA)...")

    t0 = time.time()
    gga_timestamps = []
    counts = {"GGA": 0, "VTG": 0, "RMC": 0}

    with log_path.open("w", encoding="utf-8") as f:
        f.write("# NMEA verification log\n")
        while (time.time() - t0) < seconds:
            raw = ser.readline()
            if not raw:
                continue

            try:
                line = raw.decode(errors="ignore").strip()
            except Exception:
                continue

            if not line.startswith("$"):
                continue

            f.write(line + "\n")

            if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                counts["GGA"] += 1
                gga_timestamps.append(time.time())
            elif line.startswith("$GPVTG") or line.startswith("$GNVTG"):
                counts["VTG"] += 1
            elif line.startswith("$GPRMC") or line.startswith("$GNRMC"):
                counts["RMC"] += 1

    hz = None
    if len(gga_timestamps) >= 2:
        intervals = [b - a for a, b in zip(gga_timestamps, gga_timestamps[1:])]
        if intervals:
            hz = 1.0 / statistics.mean(intervals)

    print("Counts:", counts)
    if hz is not None:
        print(f"Estimated rate (GGA): {hz:.1f} Hz")
    else:
        print("Unable to estimate rate (not enough GGA samples).")

    print(f"Verification log: {log_path.resolve()}")
    return hz, counts

# ------------------------ CLI / Main ------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Configure u-blox for 10 Hz with optimized NMEA profile (GGA/VTG 10 Hz, RMC 1 Hz)."
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port device (e.g., /dev/ttyACM0, COM3).")
    parser.add_argument("--baud", type=int, default=9600, help="Baud rate used by the serial API (required).")
    parser.add_argument("--nav-hz", type=int, default=10, help="Navigation rate in Hz. Default: 10.")
    parser.add_argument("--verify-seconds", type=float, default=6.0, help="Verification window in seconds. Default: 6.")
    parser.add_argument("--no-save", action="store_true", help="Do not attempt to persist configuration (CFG-CFG).")

    # Allow customizing the profile
    parser.add_argument("--gga-usb", type=int, default=1, help="USB rate for GGA (1 = every cycle).")
    parser.add_argument("--vtg-usb", type=int, default=1, help="USB rate for VTG (1 = every cycle).")
    parser.add_argument("--rmc-usb", type=int, default=10, help="USB rate for RMC (10 = ~1 Hz if nav=10 Hz).")
    parser.add_argument("--gsa-usb", type=int, default=0, help="USB rate for GSA (0 = off).")
    parser.add_argument("--gsv-usb", type=int, default=0, help="USB rate for GSV (0 = off).")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Opening {args.port} @ {args.baud} ...")
    print("Tip: stop gpsd before running if it is active:")
    print("  sudo systemctl stop gpsd.socket gpsd.service\n")

    with serial.Serial(args.port, args.baud, timeout=0.6) as ser:
        time.sleep(0.2)

        cfg_rate(ser, args.nav_hz)

        set_nmea_usb_profile(
            ser,
            gga_usb=args.gga_usb,
            vtg_usb=args.vtg_usb,
            rmc_usb=args.rmc_usb,
            gsa_usb=args.gsa_usb,
            gsv_usb=args.gsv_usb,
        )

        if not args.no_save:
            save_cfg(ser)

        # Verification
        ser.reset_input_buffer()
        hz, counts = verify_rate(ser, seconds=args.verify_seconds, log_path=Path("verify_10hz_gga.log"))

        print("\n== Result ==")
        target_rmc_hz = args.nav_hz / max(1, args.rmc_usb)
        print(f"Target: GGA/VTG ~{args.nav_hz} Hz, RMC ~{target_rmc_hz:.1f} Hz, GSA/GSV off.")

        if hz is not None and hz > (args.nav_hz * 0.85):
            print("Configuration applied successfully.")
        else:
            print("Warning: rate below expected. Test with open sky, stop gpsd, and retry. "
                  "Some modules only fully apply settings after unplug/replug.")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"Serial error: {e}", file=sys.stderr)
        print("Tips: check permissions (dialout group), stop gpsd "
              "(sudo systemctl stop gpsd.socket gpsd.service), and confirm the port.", file=sys.stderr)