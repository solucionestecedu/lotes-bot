"""Arma datos/distritos_sv.geojson: los 262 distritos de El Salvador (antiguos municipios) con su municipio (de los 44
de la Ley Especial para la Reestructuración Municipal, vigente desde el 1 de mayo de 2024), departamento y si están en
el AMSS (28 distritos).

Fuentes:
- Polígonos de distritos: geoBoundaries SLV ADM2 (dominio público; límites de 2007 tomados de Wikimedia Commons,
  aproximados). Solo se usan para decir en qué distrito cae un punto.
- Departamentos: geoBoundaries SLV ADM1 (CC BY-SA 2.0), solo al construir, para separar distritos con el mismo nombre
  en departamentos distintos. Sus polígonos no se guardan.
- Municipios y distritos: Ley Especial para la Reestructuración Municipal (listado publicado por Diario El Mundo).
- AMSS: 28 distritos (reforma de octubre de 2024 y acuerdo del COAMSS que sumó San Salvador Sur y Comasagua).

Uso: python herramientas/construir_distritos.py ADM2.geojson ADM1.geojson
"""
import json
import os
import sys
import unicodedata

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

LEY = {
    "Ahuachapán": {"Ahuachapán Norte": "Atiquizaya, El Refugio, San Lorenzo, Turín",
                   "Ahuachapán Centro": "Ahuachapán, Apaneca, Concepción de Ataco, Tacuba",
                   "Ahuachapán Sur": "Guaymango, Jujutla, San Francisco Menéndez, San Pedro Puxtla"},
    "San Salvador": {"San Salvador Norte": "Aguilares, El Paisnal, Guazapa",
                     "San Salvador Oeste": "Apopa, Nejapa",
                     "San Salvador Este": "Ilopango, San Martín, Soyapango, Tonacatepeque",
                     "San Salvador Centro": "Ayutuxtepeque, Mejicanos, San Salvador, Cuscatancingo, Ciudad Delgado",
                     "San Salvador Sur": "Panchimalco, Rosario de Mora, San Marcos, Santo Tomás, Santiago Texacuangos"},
    "La Libertad": {"La Libertad Norte": "Quezaltepeque, San Matías, San Pablo Tacachico",
                    "La Libertad Centro": "San Juan Opico, Ciudad Arce",
                    "La Libertad Oeste": "Colón, Jayaque, Sacacoyo, Tepecoyo, Talnique",
                    "La Libertad Este": "Antiguo Cuscatlán, Huizúcar, Nuevo Cuscatlán, San José Villanueva, Zaragoza",
                    "La Libertad Costa": "Chiltiupán, Jicalapa, La Libertad, Tamanique, Teotepeque",
                    "La Libertad Sur": "Comasagua, Santa Tecla"},
    "Chalatenango": {"Chalatenango Norte": "La Palma, Citalá, San Ignacio",
                     "Chalatenango Centro": "Nueva Concepción, Tejutla, La Reina, Agua Caliente, Dulce Nombre de María, "
                                            "El Paraíso, San Fernando, San Francisco Morazán, San Rafael, Santa Rita",
                     "Chalatenango Sur": "Chalatenango, Arcatao, Azacualpa, Comalapa, Concepción Quezaltepeque, El Carrizal, "
                                         "La Laguna, Las Vueltas, Nombre de Jesús, Nueva Trinidad, Ojos de Agua, Potonico, "
                                         "San Antonio de La Cruz, San Antonio Los Ranchos, San Francisco Lempa, "
                                         "San Isidro Labrador, San José Cancasque, San Miguel de Mercedes, "
                                         "San José Las Flores, San Luis del Carmen"},
    "Cuscatlán": {"Cuscatlán Norte": "Suchitoto, San José Guayabal, Oratorio de Concepción, San Bartolomé Perulapía, "
                                     "San Pedro Perulapán",
                  "Cuscatlán Sur": "Cojutepeque, San Rafael Cedros, Candelaria, Monte San Juan, El Carmen, San Cristóbal, "
                                   "Santa Cruz Michapa, San Ramón, El Rosario, Santa Cruz Analquito, Tenancingo"},
    "Cabañas": {"Cabañas Este": "Sensuntepeque, Victoria, Dolores, Guacotecti, San Isidro",
                "Cabañas Oeste": "Ilobasco, Tejutepeque, Jutiapa, Cinquera"},
    "La Paz": {"La Paz Oeste": "Cuyultitán, Olocuilta, San Juan Talpa, San Luis Talpa, San Pedro Masahuat, Tapalhuaca, "
                               "San Francisco Chinameca",
               "La Paz Centro": "El Rosario, Jerusalén, Mercedes La Ceiba, Paraíso de Osorio, San Antonio Masahuat, "
                                "San Emigdio, San Juan Tepezontes, San Luis La Herradura, San Miguel Tepezontes, "
                                "San Pedro Nonualco, Santa María Ostuma, Santiago Nonualco",
               "La Paz Este": "San Juan Nonualco, San Rafael Obrajuelo, Zacatecoluca"},
    "La Unión": {"La Unión Norte": "Anamorós, Bolívar, Concepción de Oriente, El Sauce, Lislique, Nueva Esparta, Pasaquina, "
                                   "Polorós, San José La Fuente, Santa Rosa de Lima",
                 "La Unión Sur": "Conchagua, El Carmen, Intipucá, La Unión, Meanguera del Golfo, San Alejo, Yayantique, "
                                 "Yucuaiquín"},
    "Usulután": {"Usulután Norte": "Santiago de María, Alegría, Berlín, Mercedes Umaña, Jucuapa, El Triunfo, Estanzuelas, "
                                   "San Buenaventura, Nueva Granada",
                 "Usulután Este": "Usulután, Jucuarán, San Dionisio, Concepción Batres, Santa María, Ozatlán, Tecapán, "
                                  "Santa Elena, California, Ereguayquín",
                 "Usulután Oeste": "Jiquilisco, Puerto El Triunfo, San Agustín, San Francisco Javier"},
    "Sonsonate": {"Sonsonate Norte": "Juayúa, Nahuizalco, Salcoatitán, Santa Catarina Masahuat",
                  "Sonsonate Centro": "Sonsonate, Sonzacate, Nahulingo, San Antonio del Monte, Santo Domingo de Guzmán",
                  "Sonsonate Este": "Izalco, Armenia, Caluco, San Julián, Cuisnahuat, Santa Isabel Ishuatán",
                  "Sonsonate Oeste": "Acajutla"},
    "Santa Ana": {"Santa Ana Norte": "Masahuat, Metapán, Santa Rosa Guachipilín, Texistepeque",
                  "Santa Ana Centro": "Santa Ana",
                  "Santa Ana Este": "Coatepeque, El Congo",
                  "Santa Ana Oeste": "Candelaria de la Frontera, Chalchuapa, El Porvenir, San Antonio Pajonal, "
                                     "San Sebastián Salitrillo, Santiago de La Frontera"},
    "San Vicente": {"San Vicente Norte": "Apastepeque, Santa Clara, San Ildefonso, San Esteban Catarina, San Sebastián, "
                                         "San Lorenzo, Santo Domingo",
                    "San Vicente Sur": "San Vicente, Guadalupe, Verapaz, Tepetitán, Tecoluca, San Cayetano Istepeque"},
    "San Miguel": {"San Miguel Norte": "Ciudad Barrios, Sesori, Nuevo Edén de San Juan, San Gerardo, San Luis de La Reina, "
                                       "Carolina, San Antonio del Mosco, Chapeltique",
                   "San Miguel Centro": "San Miguel, Comacarán, Uluazapa, Moncagua, Quelepa, Chirilagua",
                   "San Miguel Oeste": "Chinameca, Nueva Guadalupe, Lolotique, San Jorge, San Rafael Oriente, El Tránsito"},
    "Morazán": {"Morazán Norte": "Arambala, Cacaopera, Corinto, El Rosario, Joateca, Jocoaitique, Meanguera, Perquín, "
                                 "San Fernando, San Isidro, Torola",
                "Morazán Sur": "Chilanga, Delicias de Concepción, El Divisadero, Gualococti, Guatajiagua, Jocoro, "
                               "Lolotiquillo, Osicala, San Carlos, San Francisco Gotera, San Simón, Sensembra, Sociedad, "
                               "Yamabal, Yoloaiquín"},
}
AMSS_MUNICIPIOS = {"San Salvador Centro", "San Salvador Oeste", "San Salvador Este", "San Salvador Sur",
                   "La Libertad Este", "La Libertad Costa", "La Libertad Sur"}  # 7 municipios, 28 distritos

# nombres que en geoBoundaries vienen distintos
ALIAS = {"ahuachpan": "ahuachapan", "chalatenago": "chalatenango", "delgado": "ciudaddelgado",
         "concepcionataco": "concepciondeataco", "ataco": "concepciondeataco", "saningnacio": "sanignacio",
         "sanantonialosranchos": "sanantoniolosranchos", "sanfrancsicochinameca": "sanfranciscochinameca",
         "teujutepeque": "tejutepeque", "guacoteci": "guacotecti", "ciudadilobasco": "ilobasco",
         "sanemigdo": "sanemigdio"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in s if ch.isalnum())


def main(adm2, adm1, salida):
    deps = [(norm(f["properties"]["shapeName"].replace("Departamento de ", "")), shape(f["geometry"]).buffer(0))
            for f in json.load(open(adm1, encoding="utf-8"))["features"]]
    ley = {}
    for dep, munis in LEY.items():
        for muni, ds in munis.items():
            for d in ds.split(","):
                d = d.strip()
                ley[(norm(dep), norm(d))] = (dep, muni, d)
    assert len(ley) == 262, len(ley)
    grupos, sin = {}, []
    for f in json.load(open(adm2, encoding="utf-8"))["features"]:
        g = shape(f["geometry"]).buffer(0)
        dep = max(deps, key=lambda d: d[1].intersection(g).area)[0]
        n = norm(f["properties"]["shapeName"])
        n = ALIAS.get(n, n)
        clave = (dep, n)
        if clave not in ley:  # nombre corto o distinto: buscar dentro del departamento
            cands = [k for k in ley if k[0] == dep and (k[1].startswith(n) or n.startswith(k[1]) or n in k[1])]
            clave = cands[0] if len(cands) == 1 else None
        if clave is None and n != "null":  # en el borde entre departamentos: el nombre si es único en el país
            cands = [k for k in ley if k[1] == n]
            clave = cands[0] if len(cands) == 1 else None
        if clave is None:
            sin.append((dep, f["properties"]["shapeName"], round(g.area * 1e4, 3)))
            continue
        grupos.setdefault(clave, []).append(g)
    faltan = sorted(set(ley) - set(grupos))
    print("sin pareja:", sin)
    print("distritos de la ley sin polígono:", faltan)
    feats = []
    for clave, gs in sorted(grupos.items()):
        dep, muni, dist = ley[clave]
        geom = unary_union(gs).simplify(0.0004, preserve_topology=True)
        geom = shape(json.loads(json.dumps(mapping(geom), default=float)))
        feats.append({"type": "Feature", "properties": {"distrito": dist, "municipio": muni, "departamento": dep,
                                                        "amss": muni in AMSS_MUNICIPIOS},
                      "geometry": json.loads(json.dumps(mapping(geom)), parse_float=lambda v: round(float(v), 5))})
    out = {"type": "FeatureCollection",
           "fuente": "Polígonos: geoBoundaries SLV ADM2 (dominio público, límites de 2007, aproximados). Municipios: Ley "
                     "Especial para la Reestructuración Municipal (vigente desde el 1 de mayo de 2024). AMSS: 28 distritos.",
           "features": feats}
    with open(salida, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(len(feats), "distritos ·", sum(f["properties"]["amss"] for f in feats), "en el AMSS ·",
          round(os.path.getsize(salida) / 1024), "KB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else
         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datos", "distritos_sv.geojson"))
