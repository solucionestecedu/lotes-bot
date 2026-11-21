"""El terreno: vértices, tramos y verificaciones. Todo número sale de aquí, no del modelo de lenguaje."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from . import proyeccion
from .rumbos import (M2_POR_V2, area, diagnosticar_cierre, formatear_rumbo, leer_rumbo, limpiar_anillo,
                     perimetro, poligonal, rumbo_y_distancia)

TOLERANCIA_CIERRE_M = 0.10      # más de 10 cm de error de cierre = algo se leyó mal
TOLERANCIA_PRECISION = 5000     # precisión peor que 1:5,000 = sospechoso
TOLERANCIA_AREA_PCT = 0.5       # diferencia contra el área del plano


@dataclass
class Terreno:
    nombres: list[str]
    puntos: list[tuple[float, float]]           # Lambert SIRGAS-ES2007 (o locales si no está ubicado)
    tramos: list[dict] = field(default_factory=list)  # tal como vienen en el plano: {"de","a","rumbo","dist","az"}
    cierre: float = 0.0
    perimetro: float = 0.0
    area_declarada: float | None = None         # m²
    sistema_original: str = "SIRGAS"             # SIRGAS | NAD27 | local
    ubicado: bool = True
    verificaciones: list[dict] = field(default_factory=list)
    origen: str = ""

    # ------------------------------------------------------------------ medidas
    @property
    def area(self) -> float:
        return area(self.puntos)

    @property
    def precision(self) -> float:
        return self.perimetro / self.cierre if self.cierre > 1e-9 else math.inf

    def resumen(self) -> dict:
        a = self.area
        r = {
            "vertices": len(self.puntos),
            "area_m2": round(a, 2),
            "area_v2": round(a / M2_POR_V2, 2),
            "area_ha": round(a / 10000, 4),
            "perimetro_m": round(self.perimetro, 2),
            "error_cierre_m": round(self.cierre, 3),
            "precision": "perfecta" if math.isinf(self.precision) else f"1:{round(self.precision):,}",
            "cierra_bien": self.cierra_bien(),
            "ubicado": self.ubicado,
            "sistema_original": self.sistema_original,
        }
        if self.area_declarada:
            dif = (a - self.area_declarada) / self.area_declarada * 100
            r["area_declarada_m2"] = round(self.area_declarada, 2)
            r["diferencia_area_pct"] = round(dif, 2)
            r["area_coincide"] = abs(dif) <= TOLERANCIA_AREA_PCT
        if self.verificaciones:
            r["coordenadas_comprobadas"] = self.verificaciones
        if self.ubicado:
            lat, lng = proyeccion.a_latlng(*self.centroide())
            r["centro_latlng"] = [round(lat, 6), round(lng, 6)]
        return r

    def cierra_bien(self) -> bool:
        return self.cierre <= TOLERANCIA_CIERRE_M or self.precision >= TOLERANCIA_PRECISION

    def problemas(self) -> list[str]:
        out = []
        if not self.cierra_bien():
            out.append(f"La poligonal no cierra: le faltan {self.cierre:.2f} m para regresar al inicio.")
        if self.area_declarada:
            dif = (self.area - self.area_declarada) / self.area_declarada * 100
            if abs(dif) > TOLERANCIA_AREA_PCT:
                out.append(f"El área calculada ({self.area:,.2f} m²) no coincide con la del plano "
                           f"({self.area_declarada:,.2f} m²): diferencia {dif:+.2f} %.")
        for v in self.verificaciones:
            if v["diferencia_m"] > 0.10:
                out.append(f"El vértice {v['punto']} queda a {v['diferencia_m']:.2f} m de la coordenada escrita en el plano.")
        return out

    def sospechosos(self) -> list[dict]:
        """Tramos cuya distancia, si se leyó mal, explicaría el error de cierre."""
        if self.cierra_bien() or not self.tramos:
            return []
        c = [x for x in diagnosticar_cierre(self.tramos) if x["cierre_resultante"] <= TOLERANCIA_CIERRE_M]
        if self.area_declarada:  # la corrección también debe cuadrar con el área del plano
            c = [x for x in c if abs(x["area_resultante"] - self.area_declarada) / self.area_declarada * 100 <= 1.0]
        return c[:3]

    def centroide(self) -> tuple[float, float]:
        xs = [p[0] for p in self.puntos]
        ys = [p[1] for p in self.puntos]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    # ------------------------------------------------------------------ tablas
    def cuadro(self) -> list[dict]:
        """Cuadro de construcción: EST, PV, rumbo, distancia, V, X, Y."""
        n = len(self.puntos)
        filas = []
        for i in range(n):
            j = (i + 1) % n
            if self.tramos and i < len(self.tramos):
                t = self.tramos[i]
                rumbo, dist = t.get("rumbo") or formatear_rumbo(t["az"]), t["dist"]
            else:
                az, dist = rumbo_y_distancia(self.puntos[i], self.puntos[j])
                rumbo = formatear_rumbo(az)
            filas.append({"est": self.nombres[i], "pv": self.nombres[j], "rumbo": rumbo, "dist": dist,
                          "v": self.nombres[i], "x": self.puntos[i][0], "y": self.puntos[i][1]})
        return filas

    # ------------------------------------------------------------------ cambios
    def mover(self, dx: float, dy: float) -> None:
        self.puntos = [(x + dx, y + dy) for x, y in self.puntos]

    def ubicar_en(self, x: float, y: float, referencia: str = "vertice1") -> None:
        if referencia == "centro":
            cx, cy = self.centroide()
            self.mover(x - cx, y - cy)
        else:
            self.mover(x - self.puntos[0][0], y - self.puntos[0][1])
        self.ubicado = True

    def a_dict(self) -> dict:
        d = asdict(self)
        d["puntos"] = [list(p) for p in self.puntos]
        return d

    @classmethod
    def de_dict(cls, d: dict) -> "Terreno":
        d = dict(d)
        d["puntos"] = [tuple(p) for p in d["puntos"]]
        return cls(**d)


# ---------------------------------------------------------------------- constructores
def _nombres(n: int, dados: list[str] | None) -> list[str]:
    if dados and len(dados) >= n:
        return [str(x) for x in dados[:n]]
    return [str(i + 1) for i in range(n)]


def desde_tramos(tramos: list[dict], inicio: tuple[float, float] | None = None, nombres: list[str] | None = None,
                 sistema: str = "SIRGAS", area_declarada: float | None = None, origen: str = "") -> Terreno:
    """tramos: [{'az','dist', opcional 'rumbo','de','a'}]. Si no hay punto de inicio, queda sin ubicar."""
    ubicado = inicio is not None
    x0, y0 = inicio if ubicado else (0.0, 0.0)
    if ubicado and sistema == "NAD27":
        x0, y0 = proyeccion.nad27_a_sirgas(x0, y0)
    pg = poligonal((x0, y0), tramos)
    if not nombres:
        nombres = [t.get("de") for t in tramos] if all(t.get("de") for t in tramos) else None
    t = Terreno(nombres=_nombres(len(tramos), nombres), puntos=pg["puntos"], tramos=tramos, cierre=pg["cierre"],
                perimetro=pg["perimetro"], area_declarada=area_declarada, sistema_original=sistema if ubicado else "local",
                ubicado=ubicado, origen=origen)
    return t


def desde_texto_rumbos(texto: str, **kw) -> tuple[Terreno | None, list[str]]:
    from .rumbos import leer_tramos

    tramos, errores = leer_tramos(texto)
    if len(tramos) < 3:
        return None, errores or ["Se necesitan al menos 3 tramos."]
    return desde_tramos(tramos, **kw), errores


def desde_coordenadas(puntos: list[tuple[str, float, float]], sistema: str = "SIRGAS",
                      area_declarada: float | None = None, origen: str = "") -> Terreno:
    nombres = [p[0] for p in puntos]
    xy = [(float(p[1]), float(p[2])) for p in puntos]
    if sistema == "NAD27":
        xy = [proyeccion.nad27_a_sirgas(x, y) for x, y in xy]
    xy = limpiar_anillo(xy)
    tramos = []
    for i in range(len(xy)):
        az, d = rumbo_y_distancia(xy[i], xy[(i + 1) % len(xy)])
        tramos.append({"az": az, "dist": round(d, 3), "rumbo": formatear_rumbo(az)})
    return Terreno(nombres=nombres[:len(xy)], puntos=xy, tramos=tramos, cierre=0.0, perimetro=perimetro(xy),
                   area_declarada=area_declarada, sistema_original=sistema, ubicado=True, origen=origen)


def verificar_con_coordenadas(t: Terreno, conocidas: dict[str, tuple[float, float]]) -> Terreno:
    """Compara los vértices calculados con rumbos contra las coordenadas escritas en el plano.
    Si el inicio no estaba ubicado, usa la primera coordenada conocida para ubicarlo."""
    if not conocidas:
        return t
    nombres = {n.strip().upper(): i for i, n in enumerate(t.nombres)}
    pares = [(nombres[k.strip().upper()], v) for k, v in conocidas.items() if k.strip().upper() in nombres]
    if not pares:
        return t
    if not t.ubicado:
        i, (x, y) = pares[0]
        t.mover(x - t.puntos[i][0], y - t.puntos[i][1])
        t.ubicado = True
        t.sistema_original = "SIRGAS"
    t.verificaciones = [{"punto": t.nombres[i], "diferencia_m": round(math.hypot(t.puntos[i][0] - x, t.puntos[i][1] - y), 3)}
                        for i, (x, y) in pares]
    return t


def tramo_desde_texto(rumbo: str, distancia: float | None = None) -> dict:
    return leer_rumbo(f"{rumbo} {distancia}" if distancia is not None else rumbo)
