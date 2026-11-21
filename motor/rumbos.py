"""Rumbos, distancias, poligonales y geometría básica en coordenadas planas (metros)."""
from __future__ import annotations

import math
import re

M2_POR_V2 = 0.698737  # 1 vara cuadrada (vara de 0.8359 m)


class ErrorRumbo(ValueError):
    pass


def numero(texto) -> float:
    """Convierte '1,234.56', '32,50' o 32.5 en float; NaN si no se puede."""
    if isinstance(texto, (int, float)):
        return float(texto)
    t = re.sub(r"\s", "", str(texto or ""))
    if "," in t and "." in t:
        t = t.replace(",", "")
    else:
        t = t.replace(",", ".")
    try:
        return float(t) if t else math.nan
    except ValueError:
        return math.nan


def leer_rumbo(linea: str) -> dict | None:
    """Lee 'N 45°30'15" E 32.50' (acepta O/W/P para oeste) y devuelve azimut y distancia."""
    s = str(linea).upper()
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    s = re.sub(r"[°º˚'’′\"”″\-]", " ", s)
    s = re.sub(r"([A-Z])(?=\d)", r"\1 ", s)
    s = re.sub(r"(\d)(?=[A-Z])", r"\1 ", s)
    t = [x for x in re.split(r"[\s;:,]+", s) if x and not re.fullmatch(r"(M|MT|MTS|METROS)\.?", x)]
    if not t:
        return None
    if t[0] not in ("N", "S"):
        raise ErrorRumbo("debe empezar con N o S")
    i, nums, crudos = 1, [], []
    while i < len(t) and re.fullmatch(r"\d+(\.\d+)?", t[i]):
        nums.append(float(t[i]))
        crudos.append(t[i])
        i += 1
    ew = t[i] if i < len(t) else ""
    if ew not in ("E", "W", "O", "P"):
        raise ErrorRumbo("después de los grados va E (este) u O (oeste)")
    if not 1 <= len(nums) <= 3:
        raise ErrorRumbo("los grados están mal escritos")
    dist = float(t[i + 1]) if i + 1 < len(t) and re.fullmatch(r"\d+(\.\d+)?", t[i + 1]) else 0.0
    if dist <= 0:
        raise ErrorRumbo("falta la distancia al final")
    mi = nums[1] if len(nums) > 1 else 0.0
    se = nums[2] if len(nums) > 2 else 0.0
    if mi >= 60 or se >= 60:
        raise ErrorRumbo("minutos y segundos van de 0 a 59")
    a = nums[0] + mi / 60 + se / 3600
    if a > 90:
        raise ErrorRumbo("el ángulo de un rumbo no pasa de 90°")
    oeste = ew != "E"
    if t[0] == "N":
        az = 360 - a if oeste else a
    else:
        az = 180 + a if oeste else 180 - a
    ew_txt = "O" if oeste else "E"
    if len(nums) == 3 and nums[0].is_integer() and nums[1].is_integer():
        ent, _, dec = crudos[2].partition(".")
        seg = f"{int(ent):02d}" + (f".{dec}" if dec else "")
        texto = f"{t[0]} {int(nums[0]):02d}°{int(nums[1]):02d}'{seg}\" {ew_txt}"
    else:
        texto = formatear_rumbo(az % 360)
    return {"az": az % 360, "dist": dist, "angulo": a, "ns": t[0], "ew": ew_txt, "rumbo": texto}


def leer_tramos(texto: str) -> tuple[list[dict], list[str]]:
    tramos, errores = [], []
    for k, linea in enumerate(str(texto or "").splitlines(), start=1):
        if not linea.strip():
            continue
        try:
            r = leer_rumbo(linea)
            if r:
                tramos.append(r)
        except ErrorRumbo as e:
            errores.append(f"Línea {k}: {e} → \"{linea.strip()}\"")
    return tramos, errores


def poligonal(inicio: tuple[float, float], tramos: list[dict]) -> dict:
    """Recorre los tramos desde `inicio`. Devuelve vértices, error de cierre y precisión."""
    x, y = inicio
    pts = [(x, y)]
    per = 0.0
    for t in tramos:
        x += t["dist"] * math.sin(math.radians(t["az"]))
        y += t["dist"] * math.cos(math.radians(t["az"]))
        per += t["dist"]
        pts.append((x, y))
    cierre = math.hypot(x - inicio[0], y - inicio[1])
    return {
        "puntos": pts[: len(tramos)],
        "fin": (x, y),
        "cierre": cierre,
        "perimetro": per,
        "precision": per / cierre if cierre > 1e-9 else math.inf,
    }


def formatear_rumbo(az: float) -> str:
    az %= 360
    if az <= 90:
        ns, ew, a = "N", "E", az
    elif az <= 180:
        ns, ew, a = "S", "E", 180 - az
    elif az <= 270:
        ns, ew, a = "S", "O", az - 180
    else:
        ns, ew, a = "N", "O", 360 - az
    tot = round(a * 3600)
    d, tot = divmod(tot, 3600)
    m, s = divmod(tot, 60)
    return f"{ns} {d:02d}°{m:02d}'{s:02d}\" {ew}"


def rumbo_y_distancia(p, q) -> tuple[float, float]:
    dx, dy = q[0] - p[0], q[1] - p[1]
    return math.degrees(math.atan2(dx, dy)) % 360, math.hypot(dx, dy)


def area(anillo) -> float:
    s = 0.0
    n = len(anillo)
    for i in range(n):
        (x1, y1), (x2, y2) = anillo[i], anillo[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def perimetro(anillo) -> float:
    n = len(anillo)
    return sum(math.hypot(anillo[(i + 1) % n][0] - anillo[i][0], anillo[(i + 1) % n][1] - anillo[i][1]) for i in range(n))


def limpiar_anillo(anillo) -> list[tuple[float, float]]:
    out = []
    for p in anillo:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-6:
            out.append((float(p[0]), float(p[1])))
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < 1e-6:
        out.pop()
    return out


def diagnosticar_cierre(tramos: list[dict]) -> list[dict]:
    """Si la poligonal no cierra, busca qué tramo, cambiando solo su distancia, la cerraría.
    Devuelve candidatos ordenados por el cierre que lograrían (útil para pedir confirmación)."""
    base = poligonal((0.0, 0.0), tramos)
    ex, ey = base["fin"]
    out = []
    for i, t in enumerate(tramos):
        ux, uy = math.sin(math.radians(t["az"])), math.cos(math.radians(t["az"]))
        cambio = -(ex * ux + ey * uy)
        nueva = t["dist"] + cambio
        if nueva <= 0:
            continue
        prueba = [dict(tt, dist=nueva) if j == i else tt for j, tt in enumerate(tramos)]
        r = poligonal((0.0, 0.0), prueba)
        out.append({"tramo": i + 1, "distancia_leida": t["dist"], "distancia_que_cerraria": round(nueva, 3),
                    "cierre_resultante": round(r["cierre"], 3), "area_resultante": round(area(r["puntos"]), 2)})
    return sorted(out, key=lambda c: c["cierre_resultante"])
