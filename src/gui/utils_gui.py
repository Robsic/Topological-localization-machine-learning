import configparser

def parse_ini_file(filepath):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(filepath)

    if "Nodes" not in config:
        raise ValueError("Seção [Nodes] não encontrada no arquivo .ini")

    nodes = []
    for key in config["Nodes"]:
        try:
            parts = config["Nodes"][key].split(",")
            if len(parts) != 3:
                raise ValueError(f"Linha malformada em {key}: {config['Nodes'][key]}")
            node_id = int(parts[0].strip())
            lat = float(parts[1].strip())
            lon = float(parts[2].strip())
            nodes.append((node_id, lat, lon))
        except Exception as e:
            raise ValueError(f"Erro ao processar o nó {key}: {e}")
    return nodes
