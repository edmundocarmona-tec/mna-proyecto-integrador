"""
Generador del REGISTRO SINTÉTICO de necropsias (Proyecto Integrador MNA — Equipo 23).

Produce un Excel con la MISMA estructura que "REGISTRO DE PERSONAS FALLECIDAS NO IDENTIFICADAS Y NO RECLAMADAS.xlsx"
(mismos nombres de columna, filas vacías de formato, textos libres en mayúsculas), pero con contenido 100 % inventado.
Ningún valor proviene de un expediente real: los nombres son el marcador 'NOMBRE SINTETICO', las carpetas de
investigación son folios ficticios y las descripciones se arman con un vocabulario genérico.

Calibración: solo con AGREGADOS ya publicados en los Avances 3 y 4 y en el perfilado P0
(conteos, proporciones y medias por clase), nunca con filas reales.

    | Agregado (fuente)                                       | Valor objetivo          |
    |---------------------------------------------------------|-------------------------|
    | Filas del archivo / expedientes / vacías (A4, 0.1)      | 5,317 / 1,247 / 4,070    |
    | Identificados (status o nombre) (A4, 0.2)               | 355 (352 + 2 + 1)        |
    | Sexo (P0)                                               | 87 % H, 8 % M, 5 % ind.  |
    | Tipo de ingreso (P0)                                    | 86 % cadáver, 7.5 % restos óseos, 5 % fetos |
    | Edad faltante: identificados / no identificados (A4 0.4)| ~5 % / ~25 %             |
    | Estatura registrada (A4 0.4)                            | ~50 %                    |
    | Señas vacías (A4 0.4)                                   | ~35 %                    |
    | Ropa descrita, identificados / no identificados (P0)    | 71 % / 67 %, ~210 caracteres |
    | Municipio de Querétaro concentrado en la capital        | coincidencia al azar ~0.28 |

Uso:
    python generar_registro_sintetico.py            # escribe REGISTRO_SINTETICO_NECROPSIAS.xlsx junto a este script
    python generar_registro_sintetico.py --seed 7   # otra realización
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVO_SALIDA = "REGISTRO_SINTETICO_NECROPSIAS.xlsx"

COL_NOMBRE = ("EN CASO DE ESTAR IDENTIFICADO, NOMBRE QUE CORRESPONDE AL CADAVER\nDEBERA ESTABLECER AQUÍ EL NOMBRE "
              "EN EL MOMENTO EN QUE SE ACTUALICE DICHA SITUACION")
COL_SENAS = "SEÑAS PARTICULARES Y/O CARACTERISTICAS \n(CICATRICES, TATUAJES, LUNARES, MANCHAS, ETC.)"
COL_TIPO = "TIPO\n (QUÉ INGRESA A SEMEFO)"

COLUMNAS_PRINCIPALES = [
    "No", "AÑO", "FECHA DE INGRESO AL ANFITEATRO", "HORA DE NECROPSIA Y/O ANTROPOLOGICO", COL_TIPO, "SEXO",
    "AVERIGUACIÓN PREVIA O CARPETA DE INVESTIGACIÓN", "STATUS", COL_NOMBRE,
    "EN CASO DE ESTAR INHUMADO INDICAR EL PANTEÓN DONDE SE ENCUENTRA",
    "No. DE INFORME PERICIAL DE PROCESAMIENTO", "FECHA DEL INFORME", "FOLIO DE CADENA DE CUSTODIA",
    "EDAD", "ESTATURA EN CM.", "MUNICIPIO DE PROCEDENCIA", "DESCRIPCION DE ROPA QUE VESTIA", COL_SENAS,
]
# Columnas de relleno: el registro real tiene 235; aquí basta con que existan columnas administrativas cuyo
# llenado dependa del estatus (en el real, los identificados tienen más campos llenos: AUC 0.75 del conteo).
COLUMNAS_RELLENO = [f"CAMPO ADMINISTRATIVO {i:02d}" for i in range(1, 31)]

# ----------------------------------------------------------------------------- vocabulario genérico
PRENDAS_H = ["PLAYERA", "CAMISA", "CAMISETA", "PANTALON", "PANTALON DE MEZCLILLA", "SHORT", "BERMUDA", "SUDADERA",
             "CHAMARRA", "CHALECO", "TENIS", "ZAPATOS", "BOTAS", "HUARACHES", "CALCETINES", "TRUSA", "BOXER",
             "CINTURON", "GORRA"]
PRENDAS_M = ["BLUSA", "PLAYERA", "FALDA", "VESTIDO", "LEGGINS", "PANTALON DE MEZCLILLA", "SUDADERA", "SANDALIAS",
             "TENIS", "ZAPATOS", "BRASIER", "PANTALETA", "CHAMARRA", "SUETER"]
COLORES = ["NEGRO", "AZUL", "AZUL MARINO", "BLANCO", "GRIS", "ROJO", "VERDE", "CAFE", "BEIGE", "AMARILLO", "ROSA",
           "MORADO", "GUINDA", "NARANJA", "VERDE OLIVO"]
P_COLORES = np.array([18, 16, 7, 14, 12, 6, 5, 6, 4, 2, 3, 2, 2, 1, 2], float)
MARCAS = ["NIKE", "ADIDAS", "PUMA", "LEVIS", "CHARLY", "PIRMA", "FLEXI", "CAT", "POLO", "HOLLISTER", "SHEIN",
          "AMERICAN EAGLE", "SIN MARCA VISIBLE"]
TALLAS = ["CH", "M", "G", "XL", "XXL", "28", "30", "32", "34", "36"]
ESTAMPADOS = ["LETRAS", "RAYAS", "CUADROS", "UN LOGOTIPO", "FLORES", "UNA CALAVERA", "UN EQUIPO DE FUTBOL",
              "CAMUFLAJE", "UNA LEYENDA EN INGLES"]
ESTADOS = ["EN BUEN ESTADO", "EN REGULAR ESTADO", "EN MAL ESTADO", "CON MANCHAS DE TIERRA", "DESGARRADO",
           "CON MANCHAS HEMATICAS"]
REGIONES = ["ANTEBRAZO", "BRAZO", "HOMBRO", "MANO", "DORSO DE LA MANO", "MUÑECA", "PECHO", "ESPALDA", "ABDOMEN",
            "PIERNA", "MUSLO", "PANTORRILLA", "CUELLO", "TOBILLO", "REGION ESCAPULAR"]
LADOS = ["IZQUIERDO", "DERECHO", "IZQUIERDA", "DERECHA"]
MOTIVOS = ["UNA ROSA", "UNA CRUZ", "LA VIRGEN DE GUADALUPE", "UNA CALAVERA", "UNAS ALAS", "UN CORAZON",
           "UNA ESTRELLA", "LETRAS", "UNA FECHA", "UN DRAGON", "UN DISEÑO TRIBAL", "LA SANTA MUERTE", "UN RELOJ",
           "UN ROSARIO", "UN AGUILA", "UN PAYASO", "MANOS ORANDO", "UNA TELARAÑA", "UN LEON", "NUMEROS ROMANOS"]
TINTAS = ["NEGRA", "NEGRA Y ROJA", "AZUL", "MULTICOLOR", "VERDE"]
CICATRICES = ["QUIRURGICA", "LINEAL", "QUELOIDE", "IRREGULAR", "POR QUEMADURA", "ANTIGUA"]
MUNICIPIOS = ["QUERETARO", "SAN JUAN DEL RIO", "EL MARQUES", "CORREGIDORA", "PEDRO ESCOBEDO", "TEQUISQUIAPAN",
              "COLON", "AMEALCO DE BONFIL", "HUIMILPAN", "EZEQUIEL MONTES", "CADEREYTA DE MONTES", "TOLIMAN",
              "JALPAN DE SERRA", "PINAL DE AMOLES", "SAN JOAQUIN", "ARROYO SECO", "LANDA DE MATAMOROS", "PEÑAMILLER"]
P_MUN = np.array([40, 18, 9, 8, 4, 3.5, 3, 3, 2.5, 2, 2, 1.2, 1, 0.8, 0.6, 0.5, 0.5, 0.4], float)
PANTEONES = ["PANTEON MUNICIPAL", "PANTEON FORENSE", "PANTEON DE SAN JUAN DEL RIO", "PANTEON CIMATARIO"]


def _p(v):
    v = np.asarray(v, float)
    return v / v.sum()


def frase_prenda(rng, prendas):
    prenda = rng.choice(prendas)
    partes = [f"{prenda} DE COLOR {rng.choice(COLORES, p=_p(P_COLORES))}"]
    if rng.random() < 0.55:
        partes.append(f"MARCA {rng.choice(MARCAS)}")
    if rng.random() < 0.40:
        if prenda in ("TENIS", "ZAPATOS", "BOTAS", "SANDALIAS", "HUARACHES"):
            partes.append(f"DEL NUMERO {rng.integers(22, 30)}")
        else:
            partes.append(f"TALLA {rng.choice(TALLAS)}")
    if rng.random() < 0.20:
        partes.append(f"CON ESTAMPADO DE {rng.choice(ESTAMPADOS)}")
    if rng.random() < 0.40:
        partes.append(rng.choice(ESTADOS))
    return " ".join(partes)


def describir_ropa(rng, sexo):
    prendas = PRENDAS_M if sexo == "FEMENINO" else PRENDAS_H
    n = int(np.clip(rng.poisson(4.0), 2, 8))
    return ", ".join(frase_prenda(rng, prendas) for _ in range(n))


def describir_senas(rng):
    items = []
    n = int(np.clip(rng.poisson(1.6), 1, 5))
    for _ in range(n):
        u = rng.random()
        region, lado = rng.choice(REGIONES), rng.choice(LADOS)
        if u < 0.52:
            items.append(f"TATUAJE EN {region} {lado} CON LA FIGURA DE {rng.choice(MOTIVOS)} EN TINTA {rng.choice(TINTAS)}")
        elif u < 0.77:
            items.append(f"CICATRIZ {rng.choice(CICATRICES)} EN {region} {lado} DE {rng.integers(1, 20)} CM")
        elif u < 0.87:
            items.append(f"LUNAR EN {region} {lado}")
        elif u < 0.93:
            items.append(f"PERFORACION EN LOBULO DE OREJA {rng.choice(LADOS)}")
        elif u < 0.97:
            items.append("AUSENCIA DE PIEZAS DENTALES")
        else:
            items.append(f"MANCHA HIPERCROMICA EN {region} {lado}")
    return ", ".join(items)


def texto_edad(rng, edad, tipo):
    if tipo == "FETO":
        return f"PRODUCTO DE {rng.integers(18, 38)} SEMANAS"
    if edad < 2:
        return f"{max(1, int(edad * 12))} MESES"
    u = rng.random()
    if u < 0.35:
        lo = int(edad) - int(edad) % 5
        return f"{lo}-{lo + 5} AÑOS"
    if u < 0.45:
        return f"APROX. {int(edad)} AÑOS"
    return f"{int(edad)} AÑOS"


def texto_estatura(rng, cm):
    u = rng.random()
    if u < 0.005:
        return str(rng.integers(5, 15))                 # captura imposible (el A4 encontró casos así)
    if u < 0.30:
        return f"{cm / 100:.2f} M"
    if u < 0.55:
        return f"{int(cm)} CM"
    return str(int(cm))


def generar(seed=42):
    rng = np.random.default_rng(seed)
    n_exp, n_total = 1247, 5317

    # ---------------- estatus: 352 identificados con nombre, 2 sin nombre, 1 "no identificado" con nombre
    estatus = np.array(["NO IDENTIFICADO"] * n_exp, dtype=object)
    con_nombre = np.zeros(n_exp, bool)

    sexo = rng.choice(["MASCULINO", "FEMENINO", "INDETERMINADO"], n_exp, p=[0.895, 0.088, 0.017])
    tipo = rng.choice(["CADAVER", "RESTOS OSEOS", "FETO", "SEGMENTO CORPORAL"], n_exp, p=[0.865, 0.075, 0.048, 0.012])
    sexo = np.where((tipo == "FETO") & (rng.random(n_exp) < 0.4), "INDETERMINADO", sexo)
    sexo = np.where((tipo == "RESTOS OSEOS") & (rng.random(n_exp) < 0.2), "INDETERMINADO", sexo)

    # Identificación: más probable en cadáveres completos (los restos óseos y fetos casi nunca se identifican)
    w = np.where(tipo == "CADAVER", 1.0, np.where(tipo == "RESTOS OSEOS", 0.25, 0.02))
    ident = rng.choice(n_exp, 355, replace=False, p=w / w.sum())
    estatus[ident[:354]] = "IDENTIFICADO NO RECLAMADO"
    con_nombre[ident[:352]] = True                               # 352 con nombre, 2 identificados sin nombre
    otro = rng.choice(np.setdiff1d(np.arange(n_exp), ident[:354]))
    con_nombre[otro] = True                                      # 1 "NO IDENTIFICADO" con nombre (inconsistencia)
    es_ident = (estatus != "NO IDENTIFICADO") | con_nombre

    anio = np.sort(rng.choice(np.arange(2006, 2027), n_exp, p=_p(np.linspace(1, 3, 21))))
    fecha = pd.to_datetime([f"{a}-01-01" for a in anio]) + pd.to_timedelta(rng.integers(0, 365, n_exp), unit="D")

    filas = []
    for i in range(n_exp):
        s, t, idf = sexo[i], tipo[i], es_ident[i]
        r = {c: None for c in COLUMNAS_PRINCIPALES}
        r.update({"No": i + 1, "AÑO": int(anio[i]), "FECHA DE INGRESO AL ANFITEATRO": fecha[i].date(),
                  "HORA DE NECROPSIA Y/O ANTROPOLOGICO": f"{rng.integers(7, 22):02d}:{rng.choice([0, 15, 30, 45]):02d}",
                  COL_TIPO: t, "SEXO": s, "STATUS": estatus[i],
                  "AVERIGUACIÓN PREVIA O CARPETA DE INVESTIGACIÓN": f"CI/SINT/{anio[i]}/{rng.integers(10000, 99999)}",
                  COL_NOMBRE: "NOMBRE SINTETICO" if con_nombre[i] else None})
        if rng.random() < (0.8 if idf else 0.55):
            r["EN CASO DE ESTAR INHUMADO INDICAR EL PANTEÓN DONDE SE ENCUENTRA"] = rng.choice(PANTEONES)
        if rng.random() < 0.85:
            r["No. DE INFORME PERICIAL DE PROCESAMIENTO"] = f"IP-SINT-{rng.integers(1000, 9999)}"
            r["FECHA DEL INFORME"] = (fecha[i] + pd.Timedelta(days=int(rng.integers(1, 30)))).date()
        if rng.random() < 0.75:
            r["FOLIO DE CADENA DE CUSTODIA"] = f"CC-SINT-{rng.integers(100000, 999999)}"

        # Edad
        if t != "FETO":
            edad = (rng.normal(37, 14) if s != "FEMENINO" else rng.normal(35, 16))
            edad = float(np.clip(edad if rng.random() > 0.03 else rng.uniform(0.2, 14), 0.2, 92))
            if rng.random() > (0.05 if idf else 0.17 if t == "CADAVER" else 0.45):
                r["EDAD"] = texto_edad(rng, edad, t)
        elif rng.random() < 0.7:
            r["EDAD"] = texto_edad(rng, 0, t)

        # Estatura (cadáveres y restos óseos)
        if t in ("CADAVER", "RESTOS OSEOS") and rng.random() < (0.55 if idf else 0.48):
            cm = rng.normal(168, 7) if s == "MASCULINO" else rng.normal(156, 6) if s == "FEMENINO" else rng.normal(164, 9)
            r["ESTATURA EN CM."] = texto_estatura(rng, cm)

        if rng.random() < (0.72 if idf else 0.6):
            r["MUNICIPIO DE PROCEDENCIA"] = rng.choice(MUNICIPIOS, p=_p(P_MUN))

        if t != "FETO":
            if rng.random() < (0.76 if idf else 0.72 if t == "CADAVER" else 0.35):
                r["DESCRIPCION DE ROPA QUE VESTIA"] = describir_ropa(rng, s)
            if rng.random() < (0.72 if idf else 0.70 if t == "CADAVER" else 0.15):
                r[COL_SENAS] = describir_senas(rng)
            elif rng.random() < 0.35:
                r[COL_SENAS] = rng.choice(["NO PRESENTA", "SIN DATOS", "NINGUNA", "NO VALORABLE"])  # marcadores vacíos

        for c in COLUMNAS_RELLENO:
            r[c] = "DATO ADMINISTRATIVO SINTETICO" if rng.random() < (0.62 if idf else 0.55) else None
        filas.append(r)

    exp = pd.DataFrame(filas, columns=COLUMNAS_PRINCIPALES + COLUMNAS_RELLENO)
    # Filas vacías de formato: al final del archivo, como en el real. La última fila conserva un espacio en una
    # celda de relleno para que pandas no recorte las filas vacías finales al leer el Excel.
    vacias = pd.DataFrame(index=range(n_total - n_exp), columns=exp.columns, dtype=object)
    vacias.iloc[-1, -1] = " "
    return pd.concat([exp, vacias], ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--salida", default=str(Path(__file__).with_name(ARCHIVO_SALIDA)))
    args = ap.parse_args()
    df = generar(args.seed)
    df.to_excel(args.salida, index=False)
    print(f"Registro sintético escrito en {args.salida}: {len(df):,} filas x {df.shape[1]} columnas "
          f"({df['STATUS'].notna().sum():,} expedientes).")
