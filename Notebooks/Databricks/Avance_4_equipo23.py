# Databricks notebook source
# MAGIC %md
# MAGIC # Tecnológico de Monterrey
# MAGIC ## Proyecto Integrador — Equipo 23
# MAGIC ## Avance 4. Modelos alternativos
# MAGIC
# MAGIC **Cotejo de fichas de búsqueda (ante-mortem) con el libro de necropsias del SEMEFO de Querétaro (post-mortem)**
# MAGIC
# MAGIC ### Planteamiento
# MAGIC - **Pregunta:** para cada persona registrada en el libro de necropsias, ¿qué fichas de búsqueda son los candidatos más
# MAGIC   probables? (Decisión 20: consulta = necropsia, galería = fichas.)
# MAGIC - **Cotejo no supervisado** (Decisión 24): no existen pares ficha ↔ necropsia confirmados. No se entrenan modelos
# MAGIC   supervisados, no se generan datos sintéticos y se usan solo las tablas reales que ya existen.
# MAGIC - **Métrica principal: cobertura** (Decisión 25): % de necropsias cuyo mejor candidato destaca claramente sobre el
# MAGIC   resto de sus candidatos (z robusto ≥ 5). Mide que el método encuentra una señal fuerte, **no** que acierte: la
# MAGIC   confirmación es pericial.
# MAGIC - Todo se registra en **MLflow** (experimento `/Shared/proyecto_mna/avance4_cotejo`).
# MAGIC
# MAGIC ### Cómo cubre la rúbrica
# MAGIC | Criterio | Sección |
# MAGIC |---|---|
# MAGIC | Comparativa: ≥ 6 modelos individuales, ordenados por la métrica principal, ≥ 2 métricas más y tiempos | 5 y 6 |
# MAGIC | Ajuste fino de hiperparámetros de los 2 mejores | 7 |
# MAGIC | Elección argumentada del modelo final | 8 |
# MAGIC
# MAGIC ### Privacidad
# MAGIC Los datos son de personas reales. El notebook **no imprime filas ni textos**: solo conteos, métricas agregadas y
# MAGIC gráficas de agregados. En MLflow no se registran datos (sin `log_input`, sin datasets en autolog, sin tracing).
# MAGIC Los rankings se guardan en Unity Catalog (`resultados_rankings`), no en MLflow.

# COMMAND ----------

# MAGIC %pip install -q rapidfuzz rank_bm25 "splink>=4,<5" sentence-transformers optuna

# COMMAND ----------

# MAGIC %md ## 1. Parámetros

# COMMAND ----------

dbutils.widgets.text("catalog", "proyecto_mna", "Catálogo")
dbutils.widgets.text("schema", "fichas_busqueda", "Esquema")
dbutils.widgets.text("experimento", "/Shared/proyecto_mna/avance4_cotejo", "Experimento MLflow")
dbutils.widgets.dropdown("galeria", "facetas", ["facetas", "gold"], "Galería de fichas")
dbutils.widgets.text("semilla", "23", "Semilla")
dbutils.widgets.text("z_umbral", "5", "z robusto mínimo para contar una consulta como cubierta")
dbutils.widgets.text("embeddings_endpoint", "databricks-qwen3-embedding-0-6b", "Endpoint de embeddings")
dbutils.widgets.text("cross_encoder", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "Modelo cross-encoder (HF)")

# COMMAND ----------

import re
import time
import json
import posixpath
import contextlib
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from pyspark.sql import functions as F
from databricks.sdk import WorkspaceClient

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
EXPERIMENTO = dbutils.widgets.get("experimento")
GALERIA = dbutils.widgets.get("galeria")
SEMILLA = int(dbutils.widgets.get("semilla"))
Z_UMBRAL = float(dbutils.widgets.get("z_umbral"))
EMB_ENDPOINT = dbutils.widgets.get("embeddings_endpoint")
CROSS_ENCODER = dbutils.widgets.get("cross_encoder")

N_BOOTSTRAP = 1000
MIN_CANDIDATOS = 15     # con menos candidatos la escala robusta es inestable y el ruido "destaca" seguido
TOPE_Z = 10.0           # z infinito (resto empatado) se acota para promediar márgenes
MIN_TOKENS = 2          # tokens mínimos para que una necropsia sea consulta
FACETAS = ["vest", "marcas"]
PESOS = {"vest": 1.0, "marcas": 1.0}
METRICA_PRINCIPAL = "cobertura"
METRICAS_SECUNDARIAS = ["margen_mediano", "cobertura_z2", "concordancia_top10"]

def _t(nombre):
    return f"{CATALOG}.{SCHEMA}.{nombre}"

T_NECROPSIAS = _t("necropsias_enriquecidas")       # consultas
T_FICHAS_TEXTO = _t("fichas_texto_embedding")      # galería 'facetas': 60 fichas con texto por faceta
T_FICHAS_GOLD = _t("fichas_vector_search_gold")    # galería 'gold': 75 fichas, texto_canonico_fe
T_CANDIDATOS = _t("candidatos_bloqueados")
T_RANKINGS = _t("resultados_rankings")
ENTRADAS_VERSIONADAS = [T_NECROPSIAS, T_FICHAS_TEXTO, T_FICHAS_GOLD, T_CANDIDATOS]

# COMMAND ----------

# MAGIC %md ## 2. MLflow y arnés de evaluación
# MAGIC
# MAGIC Todos los modelos entregan lo mismo, un ranking `(id_consulta, id_ficha, score)`, y una sola función calcula las
# MAGIC métricas. Así la comparación es justa.
# MAGIC
# MAGIC **Evaluación sin etiquetas.** Cada método puntúa en su propia escala (BM25 no tiene tope, el coseno va de 0 a 1,
# MAGIC las reglas cuentan coincidencias), así que todo se mide **dentro de cada consulta**:
# MAGIC - z del mejor candidato = (1.º − mediana del resto) / (1.4826 × MAD del resto); si la escala es 0 se usa la
# MAGIC   desviación estándar, y si también es 0, z = ∞ cuando el 1.º supera al resto y 0 si empata.
# MAGIC - **Cobertura** = % de consultas con z ≥ 5. **Margen** = mediana de (1.º − 2.º) / escala. **Cobertura z ≥ 2** como
# MAGIC   sensibilidad. **Concordancia top-10** = Jaccard promedio del top-10 de un método con el de los demás.
# MAGIC - Calibración por simulación con scores normales puro ruido: con z ≥ 3 un método aleatorio "cubre" 16–19 % de las
# MAGIC   consultas; con z ≥ 5 y ≥ 15 candidatos, ≤ 2.5 % (0.1 % con 70 candidatos).

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
WorkspaceClient().workspace.mkdirs(posixpath.dirname(EXPERIMENTO))
mlflow.set_experiment(EXPERIMENTO)
mlflow.autolog(disable=True)
try:
    mlflow.tracing.disable()          # el tracing guardaría textos reales
except Exception:
    pass


def versiones_tablas():
    versiones = {}
    for t in ENTRADAS_VERSIONADAS:
        try:
            versiones[f"version__{t.split('.')[-1]}"] = int(spark.sql(f"DESCRIBE HISTORY {t} LIMIT 1").first()["version"])
        except Exception:
            versiones[f"version__{t.split('.')[-1]}"] = -1
    return versiones


def senal_por_consulta(rankings):
    filas = []
    for q, g in rankings.groupby("id_consulta", sort=False):
        s = np.sort(g["score"].to_numpy(dtype=float))[::-1]
        if len(s) < MIN_CANDIDATOS:
            filas.append((q, len(s), np.nan, np.nan))
            continue
        resto = s[1:]
        med = np.median(resto)
        escala = 1.4826 * np.median(np.abs(resto - med)) or resto.std()
        if escala == 0:
            z = np.inf if s[0] > med else 0.0
            margen = np.inf if s[0] > s[1] else 0.0
        else:
            z, margen = (s[0] - med) / escala, (s[0] - s[1]) / escala
        filas.append((q, len(s), z, margen))
    return pd.DataFrame(filas, columns=["id_consulta", "n_candidatos", "z_top1", "margen"])


def evaluar(rankings):
    rng = np.random.default_rng(SEMILLA)
    sp = senal_por_consulta(rankings)
    ev = sp.dropna(subset=["z_top1"])
    cubierta = (ev["z_top1"] >= Z_UMBRAL).to_numpy(dtype=float)
    reps = [cubierta[rng.integers(0, len(cubierta), len(cubierta))].mean() for _ in range(N_BOOTSTRAP)] if len(ev) else [0]
    m = {
        "cobertura": float(cubierta.mean()) if len(ev) else 0.0,
        "cobertura_ic_inf": float(np.percentile(reps, 2.5)),
        "cobertura_ic_sup": float(np.percentile(reps, 97.5)),
        "cobertura_z2": float((ev["z_top1"] >= 2).mean()) if len(ev) else 0.0,
        "margen_mediano": float(np.median(np.minimum(ev["margen"], TOPE_Z))) if len(ev) else 0.0,
        "consultas_total": int(len(sp)),
        "consultas_evaluables": int(len(ev)),
        "candidatos_por_consulta": float(sp["n_candidatos"].mean()) if len(sp) else 0.0,
    }
    return m, sp


def grafica_corrida(sp, titulo):
    z = np.minimum(sp.dropna(subset=["z_top1"])["z_top1"].to_numpy(dtype=float), TOPE_Z)
    u = np.linspace(0, TOPE_Z, 101)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(u, [(z >= x).mean() if len(z) else 0 for x in u], color="#2b6cb0")
    ax[0].axvline(Z_UMBRAL, color="#c53030", ls="--", lw=1, label=f"umbral z = {Z_UMBRAL:g}")
    ax[0].set(title="Cobertura según umbral", xlabel="z del mejor candidato", ylabel="cobertura", ylim=(0, 1.02))
    ax[0].legend(loc="upper right")
    ax[1].hist(z, bins=np.linspace(0, TOPE_Z, 41), color="#2b6cb0")
    ax[1].axvline(Z_UMBRAL, color="#c53030", ls="--", lw=1)
    ax[1].set(title=f"z del mejor candidato (acotado a {TOPE_Z:g})", xlabel="z", ylabel="consultas")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


class Corrida:
    def __init__(self, run, nombre):
        self.run, self.nombre = run, nombre
        self.tiempos = {"tiempo_entrenamiento_s": 0.0, "tiempo_inferencia_s": 0.0}
        self.metricas = None

    @contextlib.contextmanager
    def _crono(self, clave):
        t0 = time.perf_counter()
        yield
        self.tiempos[clave] += time.perf_counter() - t0

    def entrenamiento(self):
        return self._crono("tiempo_entrenamiento_s")

    def inferencia(self):
        return self._crono("tiempo_inferencia_s")

    def registrar(self, rankings, guardar_rankings=True):
        assert {"id_consulta", "id_ficha", "score"} <= set(rankings.columns), "El ranking no cumple el contrato"
        rankings = rankings[["id_consulta", "id_ficha", "score"]].fillna({"score": rankings["score"].min()})
        m, sp = evaluar(rankings)
        m.update(self.tiempos)
        m["tiempo_inferencia_ms_por_consulta"] = 1000 * self.tiempos["tiempo_inferencia_s"] / max(m["consultas_total"], 1)
        mlflow.log_metrics(m)
        mlflow.log_figure(grafica_corrida(sp, self.nombre), "graficas/cobertura_y_z.png")
        plt.close("all")
        if guardar_rankings:
            (spark.createDataFrame(rankings)
                  .selectExpr("CAST(id_consulta AS STRING) id_consulta", "CAST(id_ficha AS STRING) id_ficha",
                              "CAST(score AS DOUBLE) score")
                  .withColumn("run_id", F.lit(self.run.info.run_id)).withColumn("modelo", F.lit(self.nombre))
                  .write.mode("append").saveAsTable(T_RANKINGS))
        print(f"{self.nombre:>16}: cobertura = {m['cobertura']:.3f} [{m['cobertura_ic_inf']:.3f}–{m['cobertura_ic_sup']:.3f}]"
              f" | margen = {m['margen_mediano']:.2f} | entrenamiento = {self.tiempos['tiempo_entrenamiento_s']:.1f} s"
              f" | inferencia = {self.tiempos['tiempo_inferencia_s']:.1f} s")
        self.metricas = m
        return m


@contextlib.contextmanager
def corrida(nombre, familia, fase, params=None, tags=None):
    with mlflow.start_run(run_name=nombre, nested=mlflow.active_run() is not None) as run:
        mlflow.set_tags({"modelo": nombre, "familia": familia, "fase": fase, "supervision": "ninguna",
                         "direccion": "necropsia_a_fichas", "es_padre": "false", "galeria": GALERIA, **(tags or {})})
        mlflow.log_params({"semilla": SEMILLA, "metrica_principal": METRICA_PRINCIPAL, "z_umbral": Z_UMBRAL,
                           "min_candidatos": MIN_CANDIDATOS, "galeria": GALERIA, **(params or {})})
        mlflow.log_params(versiones_tablas())
        yield Corrida(run, nombre)


@contextlib.contextmanager
def fase_padre(fase, descripcion=""):
    with mlflow.start_run(run_name=fase) as run:
        mlflow.set_tags({"fase": fase, "es_padre": "true", "galeria": GALERIA, "mlflow.note.content": descripcion})
        yield run


def concordancia_top10(run_ids):
    r = (spark.table(T_RANKINGS).where(F.col("run_id").isin(run_ids))
              .select("run_id", "id_consulta", "id_ficha", "score").toPandas())
    r = r.sort_values(["run_id", "id_consulta", "score"], ascending=[True, True, False])
    top = r.groupby(["run_id", "id_consulta"]).head(10).groupby(["run_id", "id_consulta"])["id_ficha"].apply(frozenset)
    tops = {rid: top.xs(rid, level="run_id") for rid in run_ids if rid in top.index.get_level_values(0)}
    out = {}
    for a in tops:
        vals = [len(tops[a][q] & tops[b][q]) / len(tops[a][q] | tops[b][q])
                for b in tops if b != a for q in tops[a].index.intersection(tops[b].index)]
        out[a] = float(np.mean(vals)) if vals else np.nan
    return out


def tabla_comparativa(run_ids):
    runs = mlflow.search_runs(experiment_names=[EXPERIMENTO],
                              filter_string="tags.es_padre = 'false'")
    runs = runs[runs["run_id"].isin(run_ids)].copy()
    runs["metrics.concordancia_top10"] = runs["run_id"].map(concordancia_top10(list(runs["run_id"])))
    cols = (["tags.modelo", "tags.familia", "metrics.cobertura", "metrics.cobertura_ic_inf", "metrics.cobertura_ic_sup"]
            + [f"metrics.{m}" for m in METRICAS_SECUNDARIAS]
            + ["metrics.tiempo_entrenamiento_s", "metrics.tiempo_inferencia_ms_por_consulta", "run_id"])
    return (runs[[c for c in cols if c in runs.columns]].sort_values("metrics.cobertura", ascending=False)
                .rename(columns=lambda c: c.split(".", 1)[-1]).reset_index(drop=True))


def grafica_comparativa(t, titulo):
    t = t.iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.5 * len(t) + 1.8), sharey=True)
    ax[0].barh(t["modelo"], t["cobertura"], color="#2b6cb0",
               xerr=[t["cobertura"] - t["cobertura_ic_inf"], t["cobertura_ic_sup"] - t["cobertura"]])
    ax[0].set(xlabel=f"Cobertura (z ≥ {Z_UMBRAL:g}, IC 95 %)", xlim=(0, 1))
    ax[1].barh(t["modelo"], t["tiempo_entrenamiento_s"], color="#718096")
    ax[1].set(xlabel="Tiempo de entrenamiento (s)")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Normalización de texto (compartida por todos los modelos)
# MAGIC
# MAGIC Mayúsculas, sin acentos, sin palabras vacías y un diccionario corto de sinónimos para que "tenis" y "calzado",
# MAGIC "jeans" y "pantalón", o "negra" y "negro" cuenten igual en ambos lados. Se eliminan las palabras de relleno del
# MAGIC libro ("COLOR", "MARCA", "TALLA", "ESTADO DE CONSERVACIÓN"…, detectadas en el perfilado). **Todos los modelos reciben
# MAGIC los mismos tokens**: la diferencia está solo en cómo comparan.

# COMMAND ----------

STOP = set("""DE DEL LA LAS EL LOS Y E CON SIN EN A AL UN UNA UNOS UNAS POR PARA O U SU SUS QUE SE NO SI LE LO
TIENE TIENEN PRESENTA PRESENTAN COMO MUY MAS COLOR COLORES MARCA TALLA ESTADO CONSERVACION BUEN BUENA MAL MALA REGULAR
VISIBLE VISIBLES NINGUNA NINGUNO NINGUN TIPO PARTE CONDICIONES CONDICION ESPECIFICA ESPECIFICADA NIVEL REGION
LADO SOBRE ENTRE CADA ESTA ESTE ESTAS ESTOS DICHA DICHO OTRO OTRA OTROS OTRAS TATUAJE TATUAJES""".split())
STOP_MARCA = {"NO", "SIN", "VISIBLE", "NINGUNA", "DESCONOCIDA", "LEGIBLE", "ILEGIBLE", "APARENTE", "TALLA", "COLOR"}
SINONIMOS = {
    "PANTS": "PANTALON", "JEANS": "PANTALON", "JEAN": "PANTALON", "PANTALONES": "PANTALON",
    "TENIS": "CALZADO", "ZAPATO": "CALZADO", "ZAPATOS": "CALZADO", "BOTAS": "CALZADO", "BOTA": "CALZADO",
    "SANDALIAS": "CALZADO", "SANDALIA": "CALZADO", "HUARACHES": "CALZADO", "CHANCLAS": "CALZADO",
    "BOXER": "ROPA_INTERIOR", "TRUSA": "ROPA_INTERIOR", "CALZON": "ROPA_INTERIOR", "BRASIER": "ROPA_INTERIOR",
    "PANTALETA": "ROPA_INTERIOR", "CALZONCILLO": "ROPA_INTERIOR", "INTERIOR": "ROPA_INTERIOR",
    "CAMISETA": "PLAYERA", "PLAYERAS": "PLAYERA", "BLUSAS": "BLUSA", "CAMISAS": "CAMISA",
    "CHAQUETA": "CHAMARRA", "CHAMARRAS": "CHAMARRA", "JERSEY": "SUETER", "SWEATER": "SUETER",
    "GORRA": "GORRA_O_SOMBRERO", "SOMBRERO": "GORRA_O_SOMBRERO", "CACHUCHA": "GORRA_O_SOMBRERO",
    "BERMUDA": "SHORT", "BERMUDAS": "SHORT", "SHORTS": "SHORT", "CALCETAS": "CALCETIN", "CALCETINES": "CALCETIN",
    "AZULES": "AZUL", "GRISES": "GRIS", "MARRON": "CAFE", "CAFES": "CAFE", "GUINDA": "VINO", "CAQUI": "KAKI",
    "FIUSHA": "FUCSIA", "CELESTE": "AZUL_CLARO", "MARINO": "AZUL_MARINO",
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
    if w not in VOCAB and w.endswith("A") and w[:-1] + "O" in COLORES:
        w = w[:-1] + "O"
    return w


def _limpiar(crudo):
    return [w for w in map(_canon, crudo) if w not in STOP and (len(w) >= 3 or w.isdigit())]


def tokens(texto):
    return _limpiar(re.findall(r"[A-Z0-9]+", _sin_acentos(str(texto or "").upper())))


def elementos(texto):
    """Tokens con rol (PRENDA, COLOR, REGION, MARCA) y segmento, para reglas SQL y Splink."""
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

# MAGIC %md
# MAGIC ## 4. Datos y blocking
# MAGIC
# MAGIC - **Consultas:** `necropsias_enriquecidas` sin fetos. Faceta *vestimenta* = ropa + vestimenta + pertenencias;
# MAGIC   faceta *marcas* = descripción de tatuajes + cicatriz.
# MAGIC - **Galería** (`galeria`): `facetas` = `fichas_texto_embedding` (60 fichas; *vestimenta* = `texto_vestimenta`,
# MAGIC   *marcas* = `texto_tatuajes` + `texto_senas`). `gold` = `fichas_vector_search_gold` (75 fichas,
# MAGIC   `texto_canonico_fe`), que no describe prendas; en la comparación previa las reglas SQL dieron cobertura 0 con ella.
# MAGIC - **Blocking:** mismo sexo cuando ambos lados lo tienen (indeterminado o sin dato pasa contra todas).

# COMMAND ----------

consultas = spark.sql(f"""
    SELECT CAST(no_nec AS STRING) AS id_consulta, upper(trim(sexo_norm)) AS sexo,
           concat_ws(' ; ', ropa, descr_vestimenta, pertenencias) AS txt_vest,
           concat_ws(' ; ', descr_tatuajes, CASE WHEN upper(tiene_cicatriz) LIKE 'SI%' THEN 'cicatriz' END) AS txt_marcas
    FROM {T_NECROPSIAS}
    WHERE upper(coalesce(tipo_ingreso, '')) NOT LIKE '%FETO%'
""").toPandas()

if GALERIA == "facetas":
    galeria = spark.sql(f"""
        SELECT id_ficha, upper(trim(sexo)) AS sexo, coalesce(texto_vestimenta, '') AS txt_vest,
               concat_ws(' ; ', texto_tatuajes, texto_senas) AS txt_marcas
        FROM {T_FICHAS_TEXTO}""").toPandas()
else:
    galeria = spark.sql(f"""
        SELECT id_ficha, upper(trim(filter_sexo)) AS sexo, coalesce(texto_canonico_fe, '') AS txt_vest, '' AS txt_marcas
        FROM {T_FICHAS_GOLD}""").toPandas()

for df in (consultas, galeria):
    for f in FACETAS:
        df[f"tok_{f}"] = df[f"txt_{f}"].map(tokens)
        df[f"str_{f}"] = df[f"tok_{f}"].map(lambda t: " ".join(t).lower())

n_nec = len(consultas)
consultas = consultas[(consultas["tok_vest"].str.len() + consultas["tok_marcas"].str.len()) >= MIN_TOKENS].reset_index(drop=True)
galeria = galeria.reset_index(drop=True)

SEXOS = {"HOMBRE", "MUJER"}
pares = consultas[["id_consulta", "sexo"]].merge(galeria[["id_ficha", "sexo"]], how="cross", suffixes=("_q", "_g"))
ok = (~pares["sexo_q"].isin(SEXOS)) | (~pares["sexo_g"].isin(SEXOS)) | (pares["sexo_q"] == pares["sexo_g"])
candidatos = pares.loc[ok, ["id_consulta", "id_ficha"]].reset_index(drop=True)
(spark.createDataFrame(candidatos).withColumn("galeria", F.lit(GALERIA))
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T_CANDIDATOS))

iq = pd.Series(range(len(consultas)), index=consultas["id_consulta"])
ig = pd.Series(range(len(galeria)), index=galeria["id_ficha"])
cq, cg = iq[candidatos["id_consulta"]].to_numpy(), ig[candidatos["id_ficha"]].to_numpy()

def ranking_desde_matriz(m):
    return candidatos.assign(score=m[cq, cg])

PARAMS_COMUNES = {"peso_vestimenta": PESOS["vest"], "peso_marcas": PESOS["marcas"], "min_tokens": MIN_TOKENS,
                  "normalizador": "v1", "n_consultas": len(consultas), "n_galeria": len(galeria),
                  "n_candidatos": len(candidatos)}
print(f"Necropsias sin fetos: {n_nec} | consultas: {len(consultas)} | fichas ({GALERIA}): {len(galeria)} | "
      f"pares candidatos: {len(candidatos)} ({len(candidatos) / len(consultas):.1f} por consulta)")
for nombre, df in (("Consultas", consultas), ("Fichas", galeria)):
    print(f"{nombre} con faceta vestimenta: {(df['tok_vest'].str.len() > 0).sum()} | "
          f"con faceta marcas: {(df['tok_marcas'].str.len() > 0).sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Modelos
# MAGIC
# MAGIC Siete modelos individuales no supervisados, de lo más rígido a lo más flexible, más un baseline al azar. Todos
# MAGIC puntúan los mismos pares candidatos y combinan las facetas con la misma suma ponderada (`PESOS`). Ninguno usa
# MAGIC etiquetas: "entrenamiento" es lo que el método construye antes de consultar (índice, estimación EM, embeddings de la
# MAGIC galería, carga del modelo).
# MAGIC
# MAGIC | Modelo | Familia | Cómo compara | Entrenamiento |
# MAGIC |---|---|---|---|
# MAGIC | `B0_azar` | referencia | Score aleatorio | — |
# MAGIC | `0_sql_reglas` | determinístico | Spark SQL: misma prenda + color en el mismo segmento, misma prenda, misma marca, tatuaje en la misma región | Tablas de elementos |
# MAGIC | `1_fuzzy` | cadenas aproximadas | `token_set_ratio` (Levenshtein sobre conjuntos de palabras) | — |
# MAGIC | `2_tfidf` | léxico | TF-IDF unigramas + bigramas y coseno | Vocabulario e IDF |
# MAGIC | `3_bm25` | léxico ponderado | BM25 Okapi con las fichas como corpus | Índice BM25 |
# MAGIC | `4_splink_em` | probabilístico | Fellegi-Sunter: pesos m/u por nivel de coincidencia estimados con EM (Splink) | u por muestreo + EM |
# MAGIC | `5_embeddings` | semántica densa | Embeddings multilingües (`qwen3-embedding-0.6b`) y coseno: k-NN exacto, el mismo cálculo que hace Vector Search | Embeddings de la galería |
# MAGIC | `6_cross_encoder` | semántica de interacción | Cross-encoder multilingüe preentrenado (mMARCO), sin ajuste con nuestros datos | Carga del modelo |

# COMMAND ----------

from rapidfuzz import process, fuzz
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 0 · Reglas SQL --------------------------------------------------------------
PESOS_REGLAS = {"prenda_color": 2.0, "prenda": 1.0, "marca": 2.0, "region_marca": 2.0}
ESQ_ELEM = "id string, faceta string, seg int, token string, rol string"

def preparar_sql():
    for df, clave, vista in ((consultas, "id_consulta", "el_q"), (galeria, "id_ficha", "el_g")):
        filas = [(r[clave], f, s, t, rol) for _, r in df.iterrows() for f in FACETAS for s, t, rol in elementos(r[f"txt_{f}"])]
        spark.createDataFrame(pd.DataFrame(filas, columns=["id", "faceta", "seg", "token", "rol"]), schema=ESQ_ELEM) \
             .createOrReplaceTempView(vista)
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

def ranking_sql(pesos_reglas=PESOS_REGLAS, pesos=PESOS):
    return spark.sql(SQL_REGLAS.format(w_vest=pesos["vest"], w_marcas=pesos["marcas"], w_pc=pesos_reglas["prenda_color"],
                                       w_p=pesos_reglas["prenda"], w_m=pesos_reglas["marca"],
                                       w_r=pesos_reglas["region_marca"])).toPandas()

# --- 1 · Fuzzy ----------------------------------------------------------------------
SCORERS = {"token_set_ratio": fuzz.token_set_ratio, "token_sort_ratio": fuzz.token_sort_ratio,
           "partial_token_set_ratio": fuzz.partial_token_set_ratio, "WRatio": fuzz.WRatio}

def _sin_vacios(s, q, g):
    vq = np.array([len(x) == 0 for x in q])[:, None]
    vg = np.array([len(x) == 0 for x in g])[None, :]
    return np.where(vq | vg, 0.0, s)

def matriz_fuzzy(scorer="token_set_ratio", pesos=PESOS):
    m = np.zeros((len(consultas), len(galeria)))
    for f in FACETAS:
        q, g = list(consultas[f"str_{f}"]), list(galeria[f"str_{f}"])
        m += pesos[f] * _sin_vacios(process.cdist(q, g, scorer=SCORERS[scorer], workers=-1) / 100.0, q, g)
    return m

# --- 2 · TF-IDF ---------------------------------------------------------------------
def _ngramas(t, n):
    return t + ([f"{a}_{b}" for a, b in zip(t, t[1:])] if n >= 2 else [])

def ajustar_tfidf(ngramas=2, sublinear_tf=True, min_df=1):
    vecs = {}
    for f in FACETAS:
        docs = [_ngramas(t, ngramas) for t in list(consultas[f"tok_{f}"]) + list(galeria[f"tok_{f}"])]
        if any(docs):
            vecs[f] = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=sublinear_tf, min_df=min_df).fit(docs)
    return vecs

def matriz_tfidf(vecs, ngramas=2, pesos=PESOS):
    m = np.zeros((len(consultas), len(galeria)))
    for f, v in vecs.items():
        xq = v.transform([_ngramas(t, ngramas) for t in consultas[f"tok_{f}"]])
        xg = v.transform([_ngramas(t, ngramas) for t in galeria[f"tok_{f}"]])
        m += pesos[f] * (xq @ xg.T).toarray()
    return m

# --- 3 · BM25 -----------------------------------------------------------------------
def ajustar_bm25(k1=1.5, b=0.75):
    return {f: BM25Okapi(list(galeria[f"tok_{f}"]), k1=k1, b=b)
            for f in FACETAS if (galeria[f"tok_{f}"].str.len() > 0).any()}

def matriz_bm25(indices, pesos=PESOS):
    m = np.zeros((len(consultas), len(galeria)))
    for f, bm in indices.items():
        for i, t in enumerate(consultas[f"tok_{f}"]):
            if t:
                m[i] += pesos[f] * bm.get_scores(t)
    return m

# COMMAND ----------

# --- 4 · Fellegi-Sunter con EM (Splink) ---------------------------------------------
# Cada registro se describe con conjuntos de elementos: prendas, colores, pares prenda-color, marcas y regiones de
# tatuaje/seña. Splink compara los conjuntos por tamaño de intersección y estima con EM cuánto pesa cada nivel de
# coincidencia (m = P(nivel | mismo individuo), u = P(nivel | distinto individuo)), sin etiquetas.
import splink.comparison_library as cl
from splink import Linker, SettingsCreator, DuckDBAPI

def _conjuntos(df, clave):
    filas = []
    for _, r in df.iterrows():
        el = [(f, s, t, rol) for f in FACETAS for s, t, rol in elementos(r[f"txt_{f}"])]
        prendas = sorted({t for f, s, t, rol in el if f == "vest" and rol == "PRENDA"})
        colores = sorted({t for f, s, t, rol in el if f == "vest" and rol == "COLOR"})
        pc = sorted({f"{a}_{b}" for fa, sa, a, ra in el if fa == "vest" and ra == "PRENDA"
                     for fb, sb, b, rb in el if fb == "vest" and rb == "COLOR" and sb == sa})
        marcas = sorted({t for f, s, t, rol in el if rol == "MARCA"})
        regiones = sorted({t for f, s, t, rol in el if f == "marcas" and rol == "REGION"})
        filas.append({"unique_id": r[clave], "sexo": r["sexo"] if r["sexo"] in SEXOS else None,
                      **{k: (v or None) for k, v in [("prendas", prendas), ("colores", colores), ("prenda_color", pc),
                                                      ("marcas", marcas), ("regiones", regiones)]}})
    return pd.DataFrame(filas)

def ajustar_splink(niveles=(2, 1), max_pares_u=1e6, em_iter=25):
    q, g = _conjuntos(consultas, "id_consulta"), _conjuntos(galeria, "id_ficha")
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=[cl.ArrayIntersectAtSizes(c, list(niveles)) for c in
                     ["prendas", "colores", "prenda_color", "marcas", "regiones"]],
        blocking_rules_to_generate_predictions=["l.sexo = r.sexo", "l.sexo IS NULL", "r.sexo IS NULL"],
        probability_two_random_records_match=1.0 / len(galeria),
        max_iterations=em_iter,
        retain_intermediate_calculation_columns=False,
    )
    linker = Linker([q, g], settings, db_api=DuckDBAPI(), input_table_aliases=["necropsias", "fichas"])
    linker.training.estimate_u_using_random_sampling(max_pairs=max_pares_u, seed=SEMILLA)
    # EM con dos bloqueos distintos para que cada comparación quede estimada en al menos uno
    linker.training.estimate_parameters_using_expectation_maximisation("l.sexo = r.sexo", fix_u_probabilities=True)
    linker.training.estimate_parameters_using_expectation_maximisation(
        "array_length(list_intersect(l.prendas, r.prendas)) >= 1", fix_u_probabilities=True)
    return linker

def ranking_splink(linker):
    p = linker.inference.predict().as_pandas_dataframe()
    nec_izq = p["source_dataset_l"] == "necropsias"
    p = pd.DataFrame({"id_consulta": np.where(nec_izq, p["unique_id_l"], p["unique_id_r"]),
                      "id_ficha": np.where(nec_izq, p["unique_id_r"], p["unique_id_l"]),
                      "score": p["match_weight"]})
    rk = candidatos.merge(p, on=["id_consulta", "id_ficha"], how="left")
    return rk.fillna({"score": rk["score"].min()})

# --- 5 · Embeddings densos (Foundation Model API de Databricks) ----------------------
from mlflow.deployments import get_deploy_client

def embeber(textos, endpoint=EMB_ENDPOINT, lote=64):
    cliente = get_deploy_client("databricks")
    vecs = []
    for i in range(0, len(textos), lote):
        r = cliente.predict(endpoint=endpoint, inputs={"input": [t or "-" for t in textos[i:i + lote]]})
        vecs += [d["embedding"] for d in r["data"]]
    v = np.array(vecs, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def ajustar_embeddings(endpoint=EMB_ENDPOINT):
    return {f: embeber(list(galeria[f"str_{f}"]), endpoint) for f in FACETAS if (galeria[f"str_{f}"] != "").any()}

def matriz_embeddings(emb_galeria, endpoint=EMB_ENDPOINT, pesos=PESOS):
    m = np.zeros((len(consultas), len(galeria)))
    for f, eg in emb_galeria.items():
        q, g = list(consultas[f"str_{f}"]), list(galeria[f"str_{f}"])
        m += pesos[f] * _sin_vacios(embeber(q, endpoint) @ eg.T, q, g)
    return m

# --- 6 · Cross-encoder (preentrenado, zero-shot) --------------------------------------
from sentence_transformers import CrossEncoder

def ajustar_cross_encoder(modelo=CROSS_ENCODER):
    return CrossEncoder(modelo, max_length=256)

def matriz_cross_encoder(ce, pesos=PESOS, lote=64):
    m = np.zeros((len(consultas), len(galeria)))
    for f in FACETAS:
        q, g = list(consultas[f"str_{f}"]), list(galeria[f"str_{f}"])
        idx = [(i, j) for i, j in zip(cq, cg) if q[i] and g[j]]      # solo pares con la faceta en ambos lados
        if idx:
            s = ce.predict([(q[i], g[j]) for i, j in idx], batch_size=lote, show_progress_bar=False)
            for (i, j), v in zip(idx, s):
                m[i, j] += pesos[f] * float(v)
    return m

# COMMAND ----------

# MAGIC %md ## 6. Comparativa de los modelos
# MAGIC
# MAGIC Un run padre `comparacion` en MLflow y un run hijo por modelo. La tabla final se ordena por la métrica principal
# MAGIC (cobertura) e incluye las secundarias y los tiempos de entrenamiento e inferencia.

# COMMAND ----------

resultados = {}

with fase_padre("comparacion", f"Avance 4 — 7 modelos no supervisados + azar, galería '{GALERIA}'") as padre:
    mlflow.log_params(PARAMS_COMUNES)

    with corrida("B0_azar", "referencia", "comparacion", PARAMS_COMUNES) as c:
        with c.inferencia():
            rk = candidatos.assign(score=np.random.default_rng(SEMILLA).random(len(candidatos)))
        c.registrar(rk); resultados[c.nombre] = c.run.info.run_id

    with corrida("0_sql_reglas", "deterministico", "comparacion",
                 {**PARAMS_COMUNES, **{f"regla_{k}": v for k, v in PESOS_REGLAS.items()}}) as c:
        with c.entrenamiento():
            preparar_sql()
        with c.inferencia():
            rk = ranking_sql()
        c.registrar(rk); resultados[c.nombre] = c.run.info.run_id

    with corrida("1_fuzzy", "cadenas_aproximadas", "comparacion", {**PARAMS_COMUNES, "scorer": "token_set_ratio"}) as c:
        with c.inferencia():
            rk = ranking_desde_matriz(matriz_fuzzy())
        c.registrar(rk); resultados[c.nombre] = c.run.info.run_id

    with corrida("2_tfidf", "lexico", "comparacion", {**PARAMS_COMUNES, "ngramas": 2, "sublinear_tf": True}) as c:
        with c.entrenamiento():
            vecs = ajustar_tfidf()
        with c.inferencia():
            rk = ranking_desde_matriz(matriz_tfidf(vecs))
        c.registrar(rk); resultados[c.nombre] = c.run.info.run_id

    with corrida("3_bm25", "lexico_ponderado", "comparacion", {**PARAMS_COMUNES, "k1": 1.5, "b": 0.75}) as c:
        with c.entrenamiento():
            indices = ajustar_bm25()
        with c.inferencia():
            rk = ranking_desde_matriz(matriz_bm25(indices))
        c.registrar(rk); resultados[c.nombre] = c.run.info.run_id

    with corrida("4_splink_em", "probabilistico", "comparacion",
                 {**PARAMS_COMUNES, "niveles_interseccion": "2,1", "max_pares_u": 1e6, "em_iter": 25}) as c:
        with c.entrenamiento():
            linker = ajustar_splink()
        with c.inferencia():
            rk = ranking_splink(linker)
        c.registrar(rk); resultados[c.nombre] = c.run.info.run_id

    with corrida("5_embeddings", "semantica_densa", "comparacion", {**PARAMS_COMUNES, "endpoint": EMB_ENDPOINT}) as c:
        with c.entrenamiento():
            emb_g = ajustar_embeddings()
        with c.inferencia():
            rk = ranking_desde_matriz(matriz_embeddings(emb_g))
        c.registrar(rk); resultados[c.nombre] = c.run.info.run_id

    with corrida("6_cross_encoder", "semantica_interaccion", "comparacion", {**PARAMS_COMUNES, "modelo": CROSS_ENCODER}) as c:
        with c.entrenamiento():
            ce = ajustar_cross_encoder()
        with c.inferencia():
            rk = ranking_desde_matriz(matriz_cross_encoder(ce))
        c.registrar(rk); resultados[c.nombre] = c.run.info.run_id

    tabla = tabla_comparativa(list(resultados.values()))
    fig = grafica_comparativa(tabla, f"Comparativa de modelos — galería '{GALERIA}'")
    mlflow.log_figure(fig, "graficas/comparativa.png")
    mlflow.log_table(tabla.drop(columns=["run_id"]), "comparativa.json")     # solo métricas agregadas

# COMMAND ----------

# MAGIC %md ### Tabla comparativa (ordenada por cobertura)

# COMMAND ----------

display(tabla.drop(columns=["run_id"]))
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Ajuste fino de los 2 mejores modelos
# MAGIC
# MAGIC *(Siguiente paso: búsqueda de hiperparámetros con Optuna sobre los dos modelos con mayor cobertura, un run hijo por
# MAGIC intento bajo el run padre `ajuste_top2`.)*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Elección del modelo final
# MAGIC
# MAGIC *(Siguiente paso: comparar los dos modelos ajustados en cobertura, margen, concordancia, tiempos e
# MAGIC interpretabilidad para el perito, y justificar la elección.)*
