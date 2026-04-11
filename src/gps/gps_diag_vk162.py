#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage:

python gps_diag_vk162.py \
  --port /dev/ttyACM0 \
  --csv gps_samples.csv \
  --geojson gps_samples.geojson \
  --raw gps_raw.nmea
"""

import argparse
import csv
import json
import signal
import sys
import time
from datetime import datetime, timezone

try:
    import serial
except Exception:
    print("Error: pyserial not found. Install with: pip install pyserial", file=sys.stderr)
    sys.exit(1)

# -------------------- helpers --------------------

def now_iso_local() -> str:
    """Return local ISO timestamp with second precision."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def dm_to_deg(dm_str: str, hemisphere: str):
    """
    Convert NMEA degrees+minutes string to decimal degrees.

    dm_str:
      - Latitude:  "DDMM.MMMM"
      - Longitude: "DDDMM.MMMM"
    hemisphere: "N", "S", "E", "W"
    """
    if not dm_str or dm_str == "0":
        return None

    try:
        dm = float(dm_str)

        # Extract degrees from DM format (deg*100 + minutes)
        deg = int(dm // 100)
        minutes = dm - (deg * 100)

        value = deg + minutes / 60.0
        if hemisphere in ("S", "W"):
            value = -value

        return value
    except (ValueError, TypeError):
        return None


def parse_gga(fields):
    """
    Parse GGA sentence fields.

    Format:
    $GPGGA,time,lat,N/S,lon,E/W,fix,num_sats,hdop,alt,M,geoid,M,age,station*cs
    """
    if len(fields) < 15:
        return None

    lat = dm_to_deg(fields[2], fields[3])
    lon = dm_to_deg(fields[4], fields[5])

    try:
        fix = int(fields[6] or 0)
    except ValueError:
        fix = 0

    try:
        sats = int(fields[7] or 0)
    except ValueError:
        sats = 0

    try:
        hdop = float(fields[8]) if fields[8] else None
    except ValueError:
        hdop = None

    try:
        alt_m = float(fields[9]) if fields[9] else None
    except ValueError:
        alt_m = None

    return {"lat": lat, "lon": lon, "fix": fix, "sats": sats, "hdop": hdop, "alt_m": alt_m}


def parse_vtg(fields):
    """
    Parse VTG sentence fields and return speed in knots (if available).

    Common format:
    $GPVTG, ,T, ,M,knots,N,kmh,K,A*cs
                 ^ index 5 (knots) is commonly used
    """
    try:
        return float(fields[5]) if len(fields) > 5 and fields[5] else None
    except ValueError:
        return None


def parse_rmc(fields):
    """
    Parse RMC sentence fields and return (lat, lon, speed_knots) if status is valid.

    Format:
    $GPRMC,hhmmss.sss,A,lat,N,lon,W,speedKnots,trackAngle,date,...*cs
    """
    if len(fields) < 12:
        return None, None, None

    status = fields[2]
    if status != "A":
        return None, None, None

    lat = dm_to_deg(fields[3], fields[4])
    lon = dm_to_deg(fields[5], fields[6])

    try:
        speed_knots = float(fields[7]) if fields[7] else None
    except ValueError:
        speed_knots = None

    return lat, lon, speed_knots


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port device (e.g., /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=9600, help="Serial baud rate")
    parser.add_argument("--csv", help="Output CSV file path")
    parser.add_argument("--geojson", help="Output GeoJSON file path (FeatureCollection)")
    parser.add_argument("--raw", help="Output raw NMEA log file path")
    parser.add_argument("--no-checksum", action="store_true", help="Do not validate NMEA checksum (not implemented)")
    parser.add_argument("--flush-every", type=int, default=20, help="Flush outputs every N lines")
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    print(f"[{now_iso_local()}] Reading NMEA from {args.port} @ {args.baud}... (Ctrl+C to stop)")

    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp_local",
            "sentence",
            "lat",
            "lon",
            "alt_m",
            "hdop",
            "sats",
            "fix",
            "speed_knots",
        ])

    raw_file = open(args.raw, "w", encoding="utf-8") if args.raw else None

    # GeoJSON buffers
    points = []       # each: {"lon":..., "lat":..., "props":{...}}
    track_coords = [] # [[lon, lat], ...]

    latest_speed_knots = None
    line_count = 0
    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    while running:
        try:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line or not line.startswith("$"):
                continue

            # Raw log
            if raw_file:
                raw_file.write(f"{now_iso_local()} {line}\n")

            # Split sentence
            star_pos = line.find("*")
            body = line[1:star_pos] if star_pos != -1 else line[1:]
            fields = body.split(",")

            talker_sentence = fields[0] if fields else ""
            # e.g., "GPGGA" -> sentence "GGA"
            sentence = talker_sentence[2:] if len(talker_sentence) >= 5 else talker_sentence

            # Speed update
            if talker_sentence.endswith("VTG"):
                speed = parse_vtg(fields)
                if speed is not None:
                    latest_speed_knots = speed

            elif talker_sentence.endswith("RMC"):
                lat, lon, speed = parse_rmc(fields)
                if speed is not None:
                    latest_speed_knots = speed

                # If valid RMC provides lat/lon, record point
                if lat is not None and lon is not None:
                    props = {
                        "timestamp": now_iso_local(),
                        "src": "RMC",
                        "speed_knots": latest_speed_knots,
                    }
                    points.append({"lon": lon, "lat": lat, "props": props})
                    track_coords.append([lon, lat])

            elif talker_sentence.endswith("GGA"):
                gga = parse_gga(fields)
                if gga and gga["lat"] is not None and gga["lon"] is not None and gga["fix"] >= 1:
                    props = {
                        "timestamp": now_iso_local(),
                        "src": "GGA",
                        "alt_m": gga["alt_m"],
                        "hdop": gga["hdop"],
                        "sats": gga["sats"],
                        "fix": gga["fix"],
                        "speed_knots": latest_speed_knots,
                    }
                    points.append({"lon": gga["lon"], "lat": gga["lat"], "props": props})
                    track_coords.append([gga["lon"], gga["lat"]])

                    if csv_writer:
                        csv_writer.writerow([
                            props["timestamp"],
                            "GGA",
                            gga["lat"],
                            gga["lon"],
                            gga["alt_m"],
                            gga["hdop"],
                            gga["sats"],
                            gga["fix"],
                            latest_speed_knots,
                        ])

            line_count += 1
            if args.flush_every and (line_count % args.flush_every == 0):
                if csv_file:
                    csv_file.flush()
                if raw_file:
                    raw_file.flush()

        except KeyboardInterrupt:
            break
        except Exception as e:
            # Keep reading even if sporadic errors occur
            print(f"[WARN] {e}", file=sys.stderr)
            time.sleep(0.05)

    try:
        ser.close()
    except Exception:
        pass
    if csv_file:
        csv_file.close()
    if raw_file:
        raw_file.close()

    # ----- GeoJSON output -----
    if args.geojson:
        try:
            features = []

            # Track line (if at least 2 points)
            if len(track_coords) >= 2:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": track_coords},
                    "properties": {
                        "name": "track",
                        "points": len(track_coords),
                        "generated_at": now_iso_local(),
                    },
                })

            # Individual points (useful for inspection in geojson.io)
            for p in points:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                    "properties": p["props"],
                })

            fc = {"type": "FeatureCollection", "features": features}

            with open(args.geojson, "w", encoding="utf-8") as f:
                json.dump(fc, f, ensure_ascii=False, separators=(",", ":"))

            print(f"[OK] GeoJSON saved to: {args.geojson} (features={len(features)})")
            print("Open it at: https://geojson.io")
        except Exception as e:
            print(f"[ERROR] Failed to save GeoJSON: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        print("[INFO] --geojson not provided; no GeoJSON file generated.")


if __name__ == "__main__":
    main()
