import os
import requests
import hashlib

MAPBOX_TOKEN = "pk.eyJ1IjoibWF0ZXVzZmlsaXBlMjIiLCJhIjoiY21kYm5vaTV1MHZ5MzJzcTE1ZmQzdW55OCJ9.Al1XTYTPkJw3UIEGwFslPA"

def generate_static_map_image(nodes, folder_path=".", zoom=16):
    if not nodes:
        raise ValueError("Lista de nós vazia")

    # Gera hash exclusivo baseado nos nós
    node_str = ";".join(f"{n}-{lat:.6f}-{lon:.6f}" for n, lat, lon in sorted(nodes))
    hash_id = hashlib.md5(node_str.encode()).hexdigest()[:8]
    image_filename = f"map_static_{hash_id}.png"
    image_path = os.path.join(folder_path, image_filename)

    if os.path.exists(image_path):
        print(f"Mapa existente reutilizado: {image_path}")
        return image_path

    # Calcula centro do mapa
    latitudes = [lat for _, lat, _ in nodes]
    longitudes = [lon for _, _, lon in nodes]
    lat_center = min(latitudes) + (max(latitudes) - min(latitudes)) / 2
    lon_center = min(longitudes) + (max(longitudes) - min(longitudes)) / 2

    # Constrói os marcadores
    markers = [f"pin-l-{num}+ff0000({lon},{lat})" for num, lat, lon in nodes]
    marker_str = ",".join(markers)

    # Constrói URL
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/static/"
        f"{marker_str}/{lon_center},{lat_center},{zoom}/1100x880"
        f"?access_token={MAPBOX_TOKEN}"
    )

    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Erro ao obter imagem do mapa: {response.text}")

    # Salva imagem
    with open(image_path, "wb") as f:
        f.write(response.content)

    print(f"Mapa gerado: {image_path}")
    return image_path
