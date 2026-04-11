from geopy.distance import geodesic

def _to_float(x):
    """Converts values ​​to float; if a string containing a comma is received, replaces it with a period."""
    if isinstance(x, str):
        x = x.strip().replace(",", ".")
    return float(x)

class NodeTracker:
    """
    Accepts different node formats and internally normalizes to:
        self.nodes = {node_index: (name, num, lat, lon)}

    Accepted formats:
    - dict:
        {idx: (name, num, lat, lon)}
        {idx: (lat, lon)}
        {idx: {'lat': ..., 'lon': ..., 'name': ..., 'num' or 'id' or 'index': ...}}
    - list/tuple:
        [(name, num, lat, lon), ...]
        [(num, lat, lon), ...] # automatically generated name
        [(lat, lon), ...] # sequential num; auto name

    Note: values ​​can be strings; they will be converted to float/int.
    """
    def __init__(self, nodes, visit_threshold_m: float = 4.0):
        self.nodes = self._normalize_nodes(nodes)
        self.visit_threshold_m = visit_threshold_m   # radius to consider "visited"
        self.visited = set()                         # guard visited node_index

    def _normalize_nodes(self, nodes_input):
        """Converts various node entries into dict {node_index: (name, num, lat, lon)}."""
        norm = {}

        # --- dict case ---
        if isinstance(nodes_input, dict):
            for k, v in nodes_input.items():
                try:
                    # v can be dictated with lat/lon
                    if isinstance(v, dict):
                        # Try extracting lat/lon
                        if "lat" in v and "longitude" in v:
                            # Some configurations use 'longitude' as a key.
                            lat = _to_float(v["lat"])
                            lon = _to_float(v["longitude"])
                        else:
                            lat = _to_float(v.get("lat"))
                            lon = _to_float(v.get("lon"))
                        if lat is None or lon is None:
                            raise ValueError("lat/lon ausentes")

                        # id/num
                        if "num" in v:
                            num = int(v["num"])
                        elif "id" in v:
                            num = int(v["id"])
                        elif "index" in v:
                            num = int(v["index"])
                        else:
                            num = int(k)

                        name = v.get("name", f"NODE_{num}")
                        norm[num] = (name, num, lat, lon)

                    # v can be a tuple/list
                    elif isinstance(v, (list, tuple)):
                        if len(v) >= 4:
                            name = str(v[0])
                            num  = int(v[1])
                            lat  = _to_float(v[2])
                            lon  = _to_float(v[3])
                            norm[num] = (name, num, lat, lon)
                        elif len(v) == 3:
                            num  = int(v[0])
                            lat  = _to_float(v[1])
                            lon  = _to_float(v[2])
                            name = f"NODE_{num}"
                            norm[num] = (name, num, lat, lon)
                        elif len(v) == 2:
                            lat  = _to_float(v[0])
                            lon  = _to_float(v[1])
                            num  = int(k)
                            name = f"NODE_{num}"
                            norm[num] = (name, num, lat, lon)
                        else:
                            raise ValueError("Invalid size tuple/list")
                    else:
                        raise ValueError("node value must be dict/tuple/list")
                except Exception as e:
                    sample = f"chave={k}, valor={v}"
                    raise ValueError(f"Invalid node format in dict: {sample} — error: {e}")
            return norm

        # --- In the case of a list/tuple of items ---
        if isinstance(nodes_input, (list, tuple)):
            seq_num = 0
            for item in nodes_input:
                try:
                    if isinstance(item, dict):
                        # Try extracting lat/lon
                        if "lat" in item and "longitude" in item:
                            lat = _to_float(item["lat"])
                            lon = _to_float(item["longitude"])
                        else:
                            lat = _to_float(item.get("lat"))
                            lon = _to_float(item.get("lon"))
                        if lat is None or lon is None:
                            raise ValueError("lat/lon missing in dict")

                        if "num" in item:
                            num = int(item["num"])
                        elif "id" in item:
                            num = int(item["id"])
                        elif "index" in item:
                            num = int(item["index"])
                        else:
                            num = seq_num

                        name = item.get("name", f"NODE_{num}")
                        norm[num] = (name, num, lat, lon)
                        seq_num += 1

                    elif isinstance(item, (list, tuple)):
                        if len(item) >= 4:
                            name = str(item[0])
                            num  = int(item[1])
                            lat  = _to_float(item[2])
                            lon  = _to_float(item[3])
                            norm[num] = (name, num, lat, lon)
                        elif len(item) == 3:
                            num  = int(item[0])
                            lat  = _to_float(item[1])
                            lon  = _to_float(item[2])
                            name = f"NODE_{num}"
                            norm[num] = (name, num, lat, lon)
                        elif len(item) == 2:
                            lat  = _to_float(item[0])
                            lon  = _to_float(item[1])
                            num  = seq_num
                            name = f"NODE_{num}"
                            norm[num] = (name, num, lat, lon)
                            seq_num += 1
                        else:
                            raise ValueError("Invalid size tuple/list")
                    else:
                        raise ValueError("item must be dict/tuple/list")
                except Exception as e:
                    raise ValueError(f"Invalid node format in the list: item={item} — error: {e}")
            return norm

        # --- Case not supported ---
        raise ValueError(f"Node structure type not supported: {type(nodes_input).__name__}")

    def check_node(self, latitude: float, longitude: float):
        """
        Returns (nearest_node_index, distance_m).
        Marks as visited if distance <= visit_threshold_m.
        """
        # valida latitude/longitude
        try:
            lat = _to_float(latitude)
            lon = _to_float(longitude)
        except Exception:
            return None, None

        min_idx = None
        min_dist = None

        for idx, (_name, _num, nlat, nlon) in self.nodes.items():
            d = geodesic((lat, lon), (nlat, nlon)).meters
            if (min_dist is None) or (d < min_dist):
                min_dist = d
                min_idx = idx

        if min_idx is not None and min_dist is not None and min_dist <= self.visit_threshold_m:
            self.visited.add(min_idx)

        return min_idx, min_dist

    def all_visited(self) -> bool:
        """Returns True if all nodes have been visited (within the configured radius)."""
        return len(self.visited) >= len(self.nodes)
