import os

def create_numbered_dir(base_dir: str = "data", prefix: str = "route_images") -> str:
    """
    Create a unique output directory like:
    """
    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix + "_")
    ]

    max_id = 0
    for d in existing:
        try:
            idx = int(d.split("_")[-1])
            max_id = max(max_id, idx)
        except Exception:
            pass

    new_id = max_id + 1
    out_dir = os.path.join(base_dir, f"{prefix}_{new_id:01d}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
