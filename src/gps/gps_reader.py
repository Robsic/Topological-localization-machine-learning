import threading
import time
from collections import deque
from typing import Optional, Tuple

import serial


def dm_to_deg(dm_str: str, hemi: str) -> Optional[float]:
    """Converts NMEA values ​​from degrees+minutes (DDMM.MMMM / DDDMM.MMMM) to decimal degrees."""
    if not dm_str or dm_str == "0":
        return None
    try:
        if "." in dm_str:
            main, frac = dm_str.split(".")
            dm = float(main + "." + frac)
        else:
            dm = float(dm_str)

        # lat has 2-digit degrees (DD), lon has 3 (DDD) in the integer part.
        deg = int(dm // 100)
        minutes = dm - deg * 100
        val = deg + minutes / 60.0
        if hemi in ("S", "W"):
            val = -val
        return val
    except Exception:
        return None


def parse_gga(fields):
    # $GPGGA,time,lat,N/S,lon,E/W,fix,num_sats,hdop,alt,M,geoid,M,age,station*cs
    if len(fields) < 15:
        return None
    lat = dm_to_deg(fields[2], fields[3])
    lon = dm_to_deg(fields[4], fields[5])
    try:
        fix = int(fields[6] or 0)
    except Exception:
        fix = 0
    try:
        hdop = float(fields[8]) if fields[8] else None
    except Exception:
        hdop = None
    return {"lat": lat, "lon": lon, "fix": fix, "hdop": hdop}


def parse_rmc(fields):
    # $GPRMC,hhmmss.sss,A,lat,N,lon,W,speedKnots,trackAngle,date,...*cs
    if len(fields) < 12:
        return None
    status = fields[2]
    if status != "A":
        return None
    lat = dm_to_deg(fields[3], fields[4])
    lon = dm_to_deg(fields[5], fields[6])
    return {"lat": lat, "lon": lon}


class MovingAverageFilter2D:
    def __init__(self, window_size=5):
        self.lat_buffer = deque(maxlen=window_size)
        self.lon_buffer = deque(maxlen=window_size)

    def update(self, lat, lon) -> Tuple[float, float]:
        self.lat_buffer.append(lat)
        self.lon_buffer.append(lon)
        return (sum(self.lat_buffer) / len(self.lat_buffer),
                sum(self.lon_buffer) / len(self.lon_buffer))


class GPSReader:
    """
    Reads NMEA directly from the serial port in a background thread.
    Keeps the last valid fix and delivers it via get_coordinates().

    Parameters:

    port: e.g., "/dev/ttyACM0" (USB-ACM) or "/dev/ttyUSB0" (USB-Serial adapter)
    baud: e.g., 9600 (ignored in USB-ACM), 115200 recommended for UART with 10 Hz
    timeout: serial read time in seconds
    hdop_max: discards GGA with HDOP above this limit (if None, does not filter)
    window_size: moving average window (if 1, disables smoothing)
    """

    def __init__(self,
                 port: str = "/dev/ttyACM0",
                 baud: int = 9600,
                 timeout: float = 0.3,
                 hdop_max: Optional[float] = 5.0,
                 window_size: int = 5):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.hdop_max = hdop_max
        self.filter = MovingAverageFilter2D(window_size=window_size) if window_size >= 2 else None

        self._ser = None
        self._thread = None
        self._running = False

        # Last valid fix 
        self._last_lat = None
        self._last_lon = None
        self._last_ts = None  # time.monotonic fix
        self._lock = threading.Lock()

        # Initializes serial and thread.
        self._open_and_start()

    def _open_and_start(self):
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        # Prevents initial waste
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        """
        NMEA Read Loop:
            - Accepts GGA with fix>=1 (HDOP optional)
            - Accepts RMC status 'A'
            - Prioritizes GGA for quality; if only RMC is received, uses RMC
        """
        while self._running:
            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                try:
                    s = raw.decode("ascii", errors="ignore").strip()
                except Exception:
                    continue
                if not s.startswith("$"):
                    continue

                # Breakdown before checksum
                star = s.find("*")
                body = s[1:star] if star != -1 else s[1:]
                fields = body.split(",")
                talker = fields[0] if fields else ""

                lat = lon = None
                prefer_gga = False

                if talker.endswith("GGA"):
                    gga = parse_gga(fields)
                    if gga and gga["lat"] is not None and gga["lon"] is not None and gga["fix"] >= 1:
                        if self.hdop_max is None or gga["hdop"] is None or gga["hdop"] <= self.hdop_max:
                            lat, lon = gga["lat"], gga["lon"]
                            prefer_gga = True

                elif talker.endswith("RMC"):
                    rmc = parse_rmc(fields)
                    if rmc and rmc["lat"] is not None and rmc["lon"] is not None:
                        lat, lon = rmc["lat"], rmc["lon"]

                if lat is None or lon is None:
                    continue

                # Optional smoothing
                if self.filter:
                    lat, lon = self.filter.update(lat, lon)

                with self._lock:
                    # New fix timestamp brand
                    self._last_lat = lat
                    self._last_lon = lon
                    self._last_ts = time.monotonic()

                # If a valid GGA has been processed, you can skip to viewing others in the same cycle.
                if prefer_gga:
                    continue

            except Exception:
                # Keep the thread alive in case of occasional errors.
                time.sleep(0.02)
                continue

    def get_coordinates(self) -> Optional[Tuple[float, float]]:
        """
        Returns the last valid fix (lat, lon).
        If there is no fix yet, returns None.
        """
        with self._lock:
            if self._last_lat is None or self._last_lon is None:
                return None
            return (self._last_lat, self._last_lon)

    def close(self):
        self._running = False
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass