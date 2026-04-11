import folium
import os

def generate_map_html(nodes, output_path="map.html"):
    """
    Gera um arquivo HTML com o mapa e os marcadores dos nós.
    """
    if not nodes:
        raise ValueError("Lista de nós está vazia.")

    # Centraliza o mapa no primeiro nó
    start_coords = (nodes[0]["lat"], nodes[0]["lon"])
    m = folium.Map(location=start_coords, zoom_start=16)

    for node in nodes:
        lat, lon = node["lat"], node["lon"]
        label = f"node_{node['id']}"
        folium.Marker(location=(lat, lon), popup=label).add_to(m)

    m.save(output_path)
    return os.path.abspath(output_path)
