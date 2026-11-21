"""Coordenadas oficiales de El Salvador.

Lambert Cónica Conforme 1SP (EPSG:5399 "El Salvador Lambert"):
lat0 13.7833333, lon0 -89, k0 0.99996704, FE 500000, FN 295809.184.
- SIRGAS-ES2007 (sistema actual, desde 2007) sobre GRS80.
- Ocotepeque 1935 (época NAD27, planos viejos) sobre Clarke 1866, con towgs84 205,96,-98.
"""
from __future__ import annotations

from functools import lru_cache

from pyproj import CRS, Transformer

_BASE = "+proj=lcc +lat_1=13.7833333333333 +lat_0=13.7833333333333 +lon_0=-89 +k_0=0.99996704 +x_0=500000 +y_0=295809.184 +units=m +no_defs"
SIRGAS_LAMBERT = _BASE + " +ellps=GRS80 +towgs84=0,0,0,0,0,0,0"
NAD27_LAMBERT = _BASE + " +ellps=clrk66 +towgs84=205,96,-98,0,0,0,0"
NOMBRE_SISTEMA = "Lambert de El Salvador sobre SIRGAS-ES2007"


@lru_cache(maxsize=None)
def _t(origen: str, destino: str) -> Transformer:
    return Transformer.from_crs(CRS.from_user_input(origen), CRS.from_user_input(destino), always_xy=True)


def a_lambert(lat: float, lng: float) -> tuple[float, float]:
    x, y = _t("EPSG:4326", SIRGAS_LAMBERT).transform(lng, lat)
    return x, y


def a_latlng(x: float, y: float) -> tuple[float, float]:
    lng, lat = _t(SIRGAS_LAMBERT, "EPSG:4326").transform(x, y)
    return lat, lng


def nad27_a_latlng(x: float, y: float) -> tuple[float, float]:
    lng, lat = _t(NAD27_LAMBERT, "EPSG:4326").transform(x, y)
    return lat, lng


def latlng_a_nad27(lat: float, lng: float) -> tuple[float, float]:
    return _t("EPSG:4326", NAD27_LAMBERT).transform(lng, lat)


def nad27_a_sirgas(x: float, y: float) -> tuple[float, float]:
    return a_lambert(*nad27_a_latlng(x, y))


def en_el_salvador(x: float, y: float) -> bool:
    """Rango aproximado de coordenadas Lambert dentro del país."""
    return 380000 <= x <= 640000 and 225000 <= y <= 370000
