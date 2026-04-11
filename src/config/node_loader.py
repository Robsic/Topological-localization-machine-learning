import configparser

def load_nodes(ini_path):
    config = configparser.ConfigParser()
    config.read(ini_path)
    nodes = {}
    if "NODES" in config:
        for key in config["NODES"]:
            idx, lat, lon = map(float, config["NODES"][key].split(','))
            nodes[int(idx)] = {"lat": lat, "lon": lon, "visited": False}
    return nodes

