# Databricks notebook source
# MAGIC %md
# MAGIC Comparación de métodos rápidos (Avance 4)
# MAGIC
# MAGIC **Proyecto Integrador MNA — Cotejo ante-mortem / post-mortem (Querétaro)**
# MAGIC
# MAGIC Cotejo **no supervisado** (Decisión 24) de cada registro de necropsia contra las fichas de búsqueda, con cuatro
# MAGIC métodos rápidos y un baseline al azar. Todo queda en el experimento de MLflow del Avance 4, ordenado por
# MAGIC **cobertura** (Decisión 25).
# MAGIC
# MAGIC | Modelo | Familia | Cómo compara |
# MAGIC |---|---|---|
# MAGIC | `B0_azar` | azar | Score aleatorio: referencia de lo que da el ruido |
# MAGIC | `0_sql_reglas` | determinístico | Reglas en Spark SQL: misma prenda + color en el mismo segmento, misma prenda, misma marca, tatuaje en la misma región |
# MAGIC | `1_fuzzy` | cadenas aproximadas | `token_set_ratio` (Levenshtein sobre conjuntos de palabras) por faceta |
# MAGIC | `2_tfidf` | léxico | TF-IDF (unigramas + bigramas) y coseno por faceta |
# MAGIC | `3_bm25` | léxico ponderado | BM25 Okapi por faceta, con las fichas como corpus |
# MAGIC
# MAGIC **Datos (tablas reales existentes, nada se regenera):**
# MAGIC - Consultas: `necropsias_enriquecidas`, sin fetos. Faceta *vestimenta* = ropa + vestimenta + pertenencias;
# MAGIC   faceta *marcas* = descripción de tatuajes + cicatriz.
# MAGIC - Galería (widget `galeria`):
# MAGIC   - `facetas`: `fichas_texto_embedding` (60 fichas) — *vestimenta* = `texto_vestimenta`; *marcas* = `texto_tatuajes` + `texto_senas`.
# MAGIC   - `gold`: `fichas_vector_search_gold` (75 fichas) — todo en `texto_canonico_fe`, que no describe prendas
# MAGIC     (solo "vestía N prendas"); se deja para comprobarlo empíricamente.
# MAGIC - Blocking: mismo sexo cuando ambos lados lo tienen (indeterminado o sin dato pasa contra todas).
# MAGIC
# MAGIC **Privacidad:** el notebook no imprime filas ni textos; solo conteos y métricas agregadas.

# COMMAND ----------

# MAGIC %pip install -q rapidfuzz rank_bm25

# COMMAND ----------

# MAGIC %run ./00_config_evaluacion

# COMMAND ----------

# MAGIC %md ## 1. Parámetros

# COMMAND ----------

dbutils.widgets.dropdown("galeria", "facetas", ["facetas", "gold"], "Galería de fichas")
dbutils.widgets.text("peso_vestimenta", "1.0", "Peso faceta vestimenta")
dbutils.widgets.text("peso_marcas", "1.0", "Peso faceta marcas (tatuajes/señas)")
dbutils.widgets.text("min_tokens", "2", "Tokens mínimos para que una necropsia sea consulta")

GALERIA = dbutils.widgets.get("galeria")
PESOS = {"vest": float(dbutils.widgets.get("peso_vestimenta")), "marcas": float(dbutils.widgets.get("peso_marcas"))}
MIN_TOKENS = int(dbutils.widgets.get("min_tokens"))
FACETAS = ["vest", "marcas"]

PARAMS_COMUNES = {"galeria": GALERIA, "peso_vestimenta": PESOS["vest"], "peso_marcas": PESOS["marcas"],
                  "min_tokens": MIN_TOKENS, "normalizador": "v1"}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Normalización de texto (compartida por todos los métodos)
# MAGIC
# MAGIC Mayúsculas, sin acentos, sin palabras vacías, y un diccionario corto de sinónimos para que "tenis" y "calzado",
# MAGIC "jeans" y "pantalón", o "negra" y "negro" cuenten igual en ambos lados. Las palabras de relleno del libro
# MAGIC ("COLOR", "MARCA", "TALLA", "ESTADO DE CONSERVACIÓN"…, detectadas en el perfilado P0) se eliminan.
# MAGIC Todos los métodos reciben exactamente los mismos tokens: la diferencia está solo en cómo comparan.

# COMMAND ----------

import re
import unicodedata

STOP = set("""DE DEL LA LAS EL LOS Y E CON SIN EN A AL UN UNA UNOS UNAS POR PARA O U SU SUS QUE SE NO SI LE LO
TIENE TIENEN PRESENTA PRESENTAN COMO MUY MAS COLOR COLORES MARCA TALLA ESTADO CONSERVACION BUEN BUENA MAL MALA REGULAR
VISIBLE VISIBLES NINGUNA NINGUNO NINGUN TIPO PARTE CONDICIONES CONDICION ESPECIFICA ESPECIFICADA NIVEL REGION
LADO SOBRE ENTRE CADA ESTA ESTE ESTAS ESTOS DICHA DICHO OTRO OTRA OTROS OTRAS TATUAJE TATUAJES""".split())
STOP_MARCA = {"NO", "SIN", "VISIBLE", "NINGUNA", "DESCONOCIDA", "LEGIBLE", "ILEGIBLE", "APARENTE", "TALLA", "COLOR"}

SINONIMOS = {
    # prendas → vocabulario TipoPrenda del notebook 01
    "PANTS": "PANTALON", "JEANS": "PANTALON", "JEAN": "PANTALON", "PANTALONES": "PANTALON",
    "TENIS": "CALZADO", "ZAPATO": "CALZADO", "ZAPATOS": "CALZADO", "BOTAS": "CALZADO", "BOTA": "CALZADO",
    "SANDALIAS": "CALZADO", "SANDALIA": "CALZADO", "HUARACHES": "CALZADO", "CHANCLAS": "CALZADO",
    "BOXER": "ROPA_INTERIOR", "TRUSA": "ROPA_INTERIOR", "CALZON": "ROPA_INTERIOR", "BRASIER": "ROPA_INTERIOR",
    "PANTALETA": "ROPA_INTERIOR", "CALZONCILLO": "ROPA_INTERIOR", "INTERIOR": "ROPA_INTERIOR",
    "CAMISETA": "PLAYERA", "PLAYERAS": "PLAYERA", "BLUSAS": "BLUSA", "CAMISAS": "CAMISA",
    "CHAQUETA": "CHAMARRA", "CHAMARRAS": "CHAMARRA", "JERSEY": "SUETER", "SWEATER": "SUETER",
    "GORRA": "GORRA_O_SOMBRERO", "SOMBRERO": "GORRA_O_SOMBRERO", "CACHUCHA": "GORRA_O_SOMBRERO",
    "BERMUDA": "SHORT", "BERMUDAS": "SHORT", "SHORTS": "SHORT", "CALCETAS": "CALCETIN", "CALCETINES": "CALCETIN",
    # colores
    "AZULES": "AZUL", "GRISES": "GRIS", "MARRON": "CAFE", "CAFES": "CAFE", "GUINDA": "VINO", "CAQUI": "KAKI",
    "FIUSHA": "FUCSIA", "CELESTE": "AZUL_CLARO", "MARINO": "AZUL_MARINO",
    # regiones corporales → vocabulario Region del notebook 01
    "TORAX": "PECHO", "ESTERNAL": "PECHO", "PECTORAL": "PECHO", "ESCAPULAR": "ESPALDA", "DORSAL": "ESPALDA",
    "LUMBAR": "ESPALDA", "ABDOMINAL": "ABDOMEN", "NUCA": "CUELLO", "PANTORRILLA": "PIERNA", "DEDO": "DEDOS",
    "GLUTEA": "GLUTEO", "MUNECAS": "MUNECA", "BRAZOS": "BRAZO", "MANOS": "MANO", "PIERNAS": "PIERNA",
}
PRENDAS = {"CAMISA", "PLAYERA", "BLUSA", "SUDADERA", "CHAMARRA", "CHALECO", "SUETER", "PANTALON", "SHORT", "FALDA",
           "VESTIDO", "ROPA_INTERIOR", "CALZADO", "GORRA_O_SOMBRERO", "CALCETIN", "CINTURON", "MEZCLILLA"}
COLORES = {"NEGRO", "BLANCO", "AZUL", "AZUL_CLARO", "AZUL_MARINO", "ROJO", "VERDE", "AMARILLO", "GRIS", "CAFE",
           "BEIGE", "ROSA", "MORADO", "NARANJA", "VINO", "TURQUESA", "DORADO", "PLATEADO", "CREMA", "KAKI",
           "MOSTAZA", "LILA", "FUCSIA", "OLIVO"}
REGIONES = {"CABEZA", "CARA", "CUELLO", "HOMBRO", "PECHO", "ABDOMEN", "ESPALDA", "BRAZO", "ANTEBRAZO", "MUNECA",
            "MANO", "DEDOS", "CADERA", "GLUTEO", "MUSLO", "RODILLA", "PIERNA", "TOBILLO", "PIE"}
VOCAB = PRENDAS | COLORES | REGIONES


def _sin_acentos(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _canon(w):
    w = SINONIMOS.get(w, w)
    if w not in VOCAB and len(w) > 4 and w.endswith("ES") and w[:-2] in VOCAB:
        w = w[:-2]
    elif w not in VOCAB and len(w) > 3 and w.endswith("S"):
        w = SINONIMOS.get(w[:-1], w[:-1])
    if w not in VOCAB and w.endswith("A") and w[:-1] + "O" in COLORES:   # negra → negro
        w = w[:-1] + "O"
    return w


def _limpiar(crudo):
    out = []
    for w in crudo:
        w = _canon(w)
        if w in STOP or (len(w) < 3 and not w.isdigit()):
            continue
        out.append(w)
    return out


def tokens(texto):
    return _limpiar(re.findall(r"[A-Z0-9]+", _sin_acentos(str(texto or "").upper())))


def elementos(texto):
    """Tokens con rol (PRENDA, COLOR, REGION, MARCA) y segmento, para las reglas SQL."""
    filas = []
    for i, seg in enumerate(re.split(r"[;,.\n|]+", _sin_acentos(str(texto or "").upper()))):
        crudo = re.findall(r"[A-Z0-9]+", seg)
        for j, w in enumerate(crudo[:-1]):
            if w == "MARCA" and crudo[j + 1] not in STOP_MARCA and len(crudo[j + 1]) >= 3:
                filas.append((i, crudo[j + 1], "MARCA"))
        for w in _limpiar(crudo):
            rol = "PRENDA" if w in PRENDAS else "COLOR" if w in COLORES else "REGION" if w in REGIONES else None
            if rol:
                filas.append((i, w, rol))
    return filas

# COMMAND ----------

# MAGIC %md ## 3. Consultas, galería y blocking

# COMMAND ----------

consultas = spark.sql(f"""
    SELECT CAST(no_nec AS STRING) AS id_consulta,
           upper(trim(sexo_norm))   AS sexo,
           concat_ws(' ; ', ropa, descr_vestimenta, pertenencias) AS txt_vest,
           concat_ws(' ; ', descr_tatuajes,
                     CASE WHEN upper(tiene_cicatriz) LIKE 'SI%' THEN 'cicatriz' END) AS txt_marcas
    FROM {T_NECROPSIAS}
    WHERE upper(coalesce(tipo_ingreso, '')) NOT LIKE '%FETO%'
""").toPandas()

if GALERIA == "facetas":
    galeria = spark.sql(f"""
        SELECT id_ficha, upper(trim(sexo)) AS sexo,
               coalesce(texto_vestimenta, '') AS txt_vest,
               concat_ws(' ; ', texto_tatuajes, texto_senas) AS txt_marcas
        FROM {T_FICHAS_TEXTO}
    """).toPandas()
else:
    galeria = spark.sql(f"""
        SELECT id_ficha, upper(trim(filter_sexo)) AS sexo,
               coalesce(texto_canonico_fe, '') AS txt_vest, '' AS txt_marcas
        FROM {T_FICHAS_GOLD}
    """).toPandas()

for df in (consultas, galeria):
    for f in FACETAS:
        df[f"tok_{f}"] = df[f"txt_{f}"].map(tokens)

n_nec = len(consultas)
consultas = consultas[(consultas["tok_vest"].str.len() + consultas["tok_marcas"].str.len()) >= MIN_TOKENS]
consultas = consultas.reset_index(drop=True)
galeria = galeria.reset_index(drop=True)

SEXOS = {"HOMBRE", "MUJER"}
pares = consultas[["id_consulta", "sexo"]].merge(galeria[["id_ficha", "sexo"]], how="cross", suffixes=("_q", "_g"))
compatible = (~pares["sexo_q"].isin(SEXOS)) | (~pares["sexo_g"].isin(SEXOS)) | (pares["sexo_q"] == pares["sexo_g"])
candidatos = pares.loc[compatible, ["id_consulta", "id_ficha"]].reset_index(drop=True)

(spark.createDataFrame(candidatos).withColumn("galeria", F.lit(GALERIA))
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T_CANDIDATOS))

# Índices posicionales para pasar de matrices (consulta × ficha) a la lista de candidatos
iq = pd.Series(range(len(consultas)), index=consultas["id_consulta"])
ig = pd.Series(range(len(galeria)), index=galeria["id_ficha"])
cq = iq[candidatos["id_consulta"]].to_numpy()
cg = ig[candidatos["id_ficha"]].to_numpy()

PARAMS_COMUNES.update({"n_necropsias_sin_fetos": n_nec, "n_consultas": len(consultas), "n_galeria": len(galeria),
                       "n_candidatos": len(candidatos)})
print(f"Necropsias sin fetos: {n_nec} | consultas con ≥ {MIN_TOKENS} tokens: {len(consultas)} | "
      f"fichas en galería ({GALERIA}): {len(galeria)} | pares candidatos: {len(candidatos)}")
print("Fichas con faceta vestimenta:", int((galeria["tok_vest"].str.len() > 0).sum()),
      "| con faceta marcas:", int((galeria["tok_marcas"].str.len() > 0).sum()))
print("Consultas con faceta vestimenta:", int((consultas["tok_vest"].str.len() > 0).sum()),
      "| con faceta marcas:", int((consultas["tok_marcas"].str.len() > 0).sum()))


def ranking_desde_matriz(m: np.ndarray) -> pd.DataFrame:
    return candidatos.assign(score=m[cq, cg])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Métodos
# MAGIC
# MAGIC Cada uno se registra como un run hijo del run padre `comparacion`, con su tiempo de "entrenamiento" (lo que
# MAGIC construye antes de consultar) y de inferencia. Las facetas se combinan con la misma suma ponderada en todos.

# COMMAND ----------

import mlflow
from rapidfuzz import process, fuzz
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

TAGS = {"galeria": GALERIA}

# --- 0 · Reglas SQL -----------------------------------------------------------
PESOS_REGLAS = {"prenda_color": 2.0, "prenda": 1.0, "marca": 2.0, "region_marca": 2.0}

def _tabla_elementos(df, clave):
    filas = [(r[clave], f, seg, tok, rol) for _, r in df.iterrows() for f in FACETAS
             for seg, tok, rol in elementos(r[f"txt_{f}"])]
    return spark.createDataFrame(pd.DataFrame(filas, columns=["id", "faceta", "seg", "token", "rol"]),
                                 schema="id string, faceta string, seg int, token string, rol string")

def correr_sql():
    _tabla_elementos(consultas, "id_consulta").createOrReplaceTempView("el_q")
    _tabla_elementos(galeria, "id_ficha").createOrReplaceTempView("el_g")
    spark.createDataFrame(candidatos).createOrReplaceTempView("cand")


SQL_REGLAS = """
WITH pc_q AS (SELECT DISTINCT a.id, a.token AS prenda, b.token AS color FROM el_q a JOIN el_q b
              ON a.id = b.id AND a.faceta = b.faceta AND a.seg = b.seg
              WHERE a.faceta = 'vest' AND a.rol = 'PRENDA' AND b.rol = 'COLOR'),
     pc_g AS (SELECT DISTINCT a.id, a.token AS prenda, b.token AS color FROM el_g a JOIN el_g b
              ON a.id = b.id AND a.faceta = b.faceta AND a.seg = b.seg
              WHERE a.faceta = 'vest' AND a.rol = 'PRENDA' AND b.rol = 'COLOR'),
     un_q AS (SELECT DISTINCT id, faceta, rol, token FROM el_q),
     un_g AS (SELECT DISTINCT id, faceta, rol, token FROM el_g),
     r_pc AS (SELECT c.id_consulta, c.id_ficha, count(*) AS n FROM cand c
              JOIN pc_q q ON q.id = c.id_consulta
              JOIN pc_g g ON g.id = c.id_ficha AND g.prenda = q.prenda AND g.color = q.color
              GROUP BY c.id_consulta, c.id_ficha),
     r_rol AS (SELECT c.id_consulta, c.id_ficha,
                      sum(CASE WHEN q.faceta = 'vest' AND q.rol = 'PRENDA' THEN 1 ELSE 0 END) AS n_prenda,
                      sum(CASE WHEN q.faceta = 'vest' AND q.rol = 'MARCA' THEN 1 ELSE 0 END) AS n_marca,
                      sum(CASE WHEN q.faceta = 'marcas' AND q.rol = 'REGION' THEN 1 ELSE 0 END) AS n_region
               FROM cand c
               JOIN un_q q ON q.id = c.id_consulta
               JOIN un_g g ON g.id = c.id_ficha AND g.faceta = q.faceta AND g.rol = q.rol AND g.token = q.token
               GROUP BY c.id_consulta, c.id_ficha)
SELECT c.id_consulta, c.id_ficha,
       {w_vest} * ({w_pc} * coalesce(pc.n, 0) + {w_p} * coalesce(r.n_prenda, 0) + {w_m} * coalesce(r.n_marca, 0))
     + {w_marcas} * {w_r} * coalesce(r.n_region, 0) AS score
FROM cand c
LEFT JOIN r_pc pc ON pc.id_consulta = c.id_consulta AND pc.id_ficha = c.id_ficha
LEFT JOIN r_rol r ON r.id_consulta = c.id_consulta AND r.id_ficha = c.id_ficha
"""

# --- 1 · Fuzzy ------------------------------------------------------------------
def matriz_fuzzy(scorer=fuzz.token_set_ratio):
    m = np.zeros((len(consultas), len(galeria)))
    for f in FACETAS:
        q = [" ".join(t) for t in consultas[f"tok_{f}"]]
        g = [" ".join(t) for t in galeria[f"tok_{f}"]]
        s = process.cdist(q, g, scorer=scorer, workers=-1) / 100.0
        vacios = np.outer([len(x) == 0 for x in q], np.ones(len(g))) + np.outer(np.ones(len(q)), [len(x) == 0 for x in g])
        m += PESOS[f] * np.where(vacios > 0, 0.0, s)
    return m

# --- 2 · TF-IDF -----------------------------------------------------------------
def _con_bigramas(t):
    return t + [f"{a}_{b}" for a, b in zip(t, t[1:])]

def ajustar_tfidf():
    vecs = {}
    for f in FACETAS:
        docs = [_con_bigramas(t) for t in list(consultas[f"tok_{f}"]) + list(galeria[f"tok_{f}"])]
        if not any(docs):
            continue
        vecs[f] = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True, min_df=1).fit(docs)
    return vecs

def matriz_tfidf(vecs):
    m = np.zeros((len(consultas), len(galeria)))
    for f, v in vecs.items():
        xq = v.transform([_con_bigramas(t) for t in consultas[f"tok_{f}"]])
        xg = v.transform([_con_bigramas(t) for t in galeria[f"tok_{f}"]])
        m += PESOS[f] * (xq @ xg.T).toarray()
    return m

# --- 3 · BM25 -------------------------------------------------------------------
BM25_K1, BM25_B = 1.5, 0.75

def ajustar_bm25():
    return {f: BM25Okapi(list(galeria[f"tok_{f}"]), k1=BM25_K1, b=BM25_B)
            for f in FACETAS if (galeria[f"tok_{f}"].str.len() > 0).any()}

def matriz_bm25(indices):
    m = np.zeros((len(consultas), len(galeria)))
    for f, bm in indices.items():
        for i, t in enumerate(consultas[f"tok_{f}"]):
            if t:
                m[i] += PESOS[f] * bm.get_scores(t)
    return m

# COMMAND ----------

# MAGIC %md ## 5. Ejecución y registro en MLflow

# COMMAND ----------

with fase_padre("comparacion", f"Métodos rápidos sobre galería '{GALERIA}' (notebook 20)") as padre:
    mlflow.set_tags(TAGS)
    mlflow.log_params(PARAMS_COMUNES)

    with corrida("B0_azar", familia="azar", fase="comparacion", params=PARAMS_COMUNES, tags=TAGS) as c:
        with c.inferencia():
            rng = np.random.default_rng(SEMILLA)
            rk = candidatos.assign(score=rng.random(len(candidatos)))
        c.registrar(rk)

    with corrida("0_sql_reglas", familia="deterministico", fase="comparacion",
                 params={**PARAMS_COMUNES, **{f"regla_{k}": v for k, v in PESOS_REGLAS.items()}}, tags=TAGS) as c:
        with c.entrenamiento():
            correr_sql()
        with c.inferencia():
            rk = spark.sql(SQL_REGLAS.format(w_vest=PESOS["vest"], w_marcas=PESOS["marcas"],
                                             w_pc=PESOS_REGLAS["prenda_color"], w_p=PESOS_REGLAS["prenda"],
                                             w_m=PESOS_REGLAS["marca"], w_r=PESOS_REGLAS["region_marca"])).toPandas()
        c.registrar(rk)

    with corrida("1_fuzzy", familia="cadenas_aproximadas", fase="comparacion",
                 params={**PARAMS_COMUNES, "scorer": "token_set_ratio"}, tags=TAGS) as c:
        with c.inferencia():
            rk = ranking_desde_matriz(matriz_fuzzy())
        c.registrar(rk)

    with corrida("2_tfidf", familia="lexico", fase="comparacion",
                 params={**PARAMS_COMUNES, "ngramas": "1-2", "sublinear_tf": True}, tags=TAGS) as c:
        with c.entrenamiento():
            vecs = ajustar_tfidf()
        with c.inferencia():
            rk = ranking_desde_matriz(matriz_tfidf(vecs))
        c.registrar(rk)

    with corrida("3_bm25", familia="lexico_ponderado", fase="comparacion",
                 params={**PARAMS_COMUNES, "k1": BM25_K1, "b": BM25_B}, tags=TAGS) as c:
        with c.entrenamiento():
            indices = ajustar_bm25()
        with c.inferencia():
            rk = ranking_desde_matriz(matriz_bm25(indices))
        c.registrar(rk)

    # Comparativa (solo agregados), también guardada en el run padre
    tabla = tabla_comparativa("comparacion", filtro=f"tags.galeria = '{GALERIA}'")
    fig = grafica_comparativa(tabla, f"Métodos rápidos — galería '{GALERIA}'")
    mlflow.log_figure(fig, "graficas/comparativa.png")
    mlflow.log_table(tabla.drop(columns=["run_id"]), "comparativa.json")   # métricas agregadas, sin datos

# COMMAND ----------

# MAGIC %md ## 6. Tabla comparativa (ordenada por cobertura)

# COMMAND ----------

display(tabla)
display(fig)