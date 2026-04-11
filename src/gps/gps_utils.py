import math

def compute_heading(coord1, coord2):
    """Calculates the heading (direction angle) between two GPS points in degrees [0, 360)."""
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def angular_diff_deg(a, b):
    """Minimum circular angular difference in degrees (0 to 180)."""
    d = abs((a - b + 180) % 360 - 180)
    return d

def circular_mean_deg(values_deg):
    """Mean circular angle in degrees."""
    if not values_deg:
        return None
    s = sum(math.sin(math.radians(v)) for v in values_deg)
    c = sum(math.cos(math.radians(v)) for v in values_deg)
    return (math.degrees(math.atan2(s, c)) + 360) % 360