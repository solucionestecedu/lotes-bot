"""Paquete de calles, ríos y construcciones de El Salvador sacado de OpenStreetMap (descarga de Geofabrik).

Se corre al construir la imagen del bot (Dockerfile), así los datos se refrescan en cada publicación y el bot no
depende de servidores de afuera mientras atiende:

    python herramientas/construir_osm.py datos/osm_sv.npz [archivo.osm.pbf]

Guarda en Lambert (SIRGAS) las líneas simplificadas a 1 m de calles, senderos, ríos, quebradas, canales y cuerpos de
agua (también los lagos armados como multipolígono), con su tipo y nombre, y cuántas construcciones hay en cada cuadro
de 100 m. Datos © colaboradores de OpenStreetMap, licencia ODbL (el paquete derivado queda bajo la misma licencia).
Si algo falla deja `<salida>.falta` con el error y el bot funciona igual, solo que sin calles ni ríos del mapa.
"""
from __future__ import annotations

import datetime
import os
import sys
import tempfile
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

URL = "https://download.geofabrik.de/central-america/el-salvador-latest.osm.pbf"


def leer(pbf: str):
    """(vías [(tipo, nombre, [(lon, lat)…])], construcciones [(lon, lat)…]) del archivo de OpenStreetMap."""
    import osmium

    from servicios.osm import TIPOS

    validos = set(TIPOS)
    vias, edif = [], []
    fp = (osmium.FileProcessor(pbf, osmium.osm.WAY | osmium.osm.RELATION | osmium.osm.NODE)
          .with_locations()
          .with_areas(osmium.filter.TagFilter(("natural", "water")))
          .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY | osmium.osm.AREA))
          .with_filter(osmium.filter.KeyFilter("highway", "waterway", "natural", "building")))
    for o in fp:
        t = o.tags
        if o.is_area():
            if t.get("natural") != "water":
                continue
            for anillo in o.outer_rings():
                pts = [(n.lon, n.lat) for n in anillo if n.location.valid()]
                if len(pts) >= 4:
                    vias.append(("natural=water", t.get("name"), pts))
            continue
        if "building" in t:
            for n in o.nodes:
                if n.location.valid():
                    edif.append((n.lon, n.lat))
                    break
            continue
        if "highway" in t:
            tipo = "highway=" + t["highway"]
        elif "waterway" in t:
            tipo = "waterway=" + t["waterway"]
        else:
            continue  # natural=water ya sale como área (cerrada o multipolígono)
        if tipo not in validos:
            continue
        pts = [(n.lon, n.lat) for n in o.nodes if n.location.valid()]
        if len(pts) >= 2:
            vias.append((tipo, t.get("name"), pts))
    return vias, edif


def construir(pbf: str, salida: str) -> dict:
    from pyproj import Transformer

    from motor.proyeccion import SIRGAS_LAMBERT
    from servicios import osm

    t0 = time.time()
    vias, edif = leer(pbf)
    t1 = time.time()
    tr = Transformer.from_crs("EPSG:4326", SIRGAS_LAMBERT, always_xy=True)
    lam = []
    for tipo, nombre, pts in vias:
        a = np.asarray(pts, dtype=float)
        x, y = tr.transform(a[:, 0], a[:, 1])
        lam.append((tipo, nombre, np.column_stack([x, y])))
    e = None
    if edif:
        a = np.asarray(edif, dtype=float)
        x, y = tr.transform(a[:, 0], a[:, 1])
        e = np.column_stack([x, y])
    paquete = osm.empacar(lam, e, fecha=datetime.date.today().isoformat())
    osm.guardar(paquete, salida)
    import resource
    meta = dict(paquete["meta"], lectura_s=round(t1 - t0, 1), total_s=round(time.time() - t0, 1),
                memoria_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    meta.pop("tipos", None)
    return meta


def bajar(destino: str) -> None:
    req = urllib.request.Request(URL, headers={"User-Agent": "Lotes-Bot/1.0 (paquete de OpenStreetMap)"})
    with urllib.request.urlopen(req, timeout=300) as r, open(destino, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)


def main():
    salida = sys.argv[1] if len(sys.argv) > 1 else "datos/osm_sv.npz"
    os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
    pbf = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        if pbf is None:
            pbf = os.path.join(tempfile.mkdtemp(), "el-salvador.osm.pbf")
            bajar(pbf)
            print(f"Descargado {URL} ({os.path.getsize(pbf) / 1e6:.1f} MB)")
        meta = construir(pbf, salida)
        print(f"Paquete OSM listo: {salida} ({os.path.getsize(salida) / 1e6:.1f} MB)", meta)
    except Exception as ex:  # noqa: BLE001
        print("No se pudo construir el paquete OSM:", repr(ex))
        with open(salida + ".falta", "w", encoding="utf-8") as f:
            f.write(repr(ex))


if __name__ == "__main__":
    main()
