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


_VERSIONES = {}

def versiones_tablas():
    """Versión Delta de cada entrada. Se calcula una sola vez por sesión (después de escribir los candidatos)."""
    if _VERSIONES:
        return _VERSIONES
    versiones = _VERSIONES
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
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
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
VARIANTES_BM25 = {"okapi": BM25Okapi, "l": BM25L, "plus": BM25Plus}

def ajustar_bm25(k1=1.5, b=0.75, variante="okapi"):
    return {f: VARIANTES_BM25[variante](list(galeria[f"tok_{f}"]), k1=k1, b=b)
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

def ajustar_splink(niveles=(2, 1), max_pares_u=1e6, em_iter=25, factor_previa=1.0):
    q, g = _conjuntos(consultas, "id_consulta"), _conjuntos(galeria, "id_ficha")
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=[cl.ArrayIntersectAtSizes(c, list(niveles)) for c in
                     ["prendas", "colores", "prenda_color", "marcas", "regiones"]],
        blocking_rules_to_generate_predictions=["l.sexo = r.sexo", "l.sexo IS NULL", "r.sexo IS NULL"],
        probability_two_random_records_match=min(factor_previa / len(galeria), 0.5),
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

def _predict_con_reintentos(cliente, endpoint, entradas, intentos=8):
    """El endpoint de pago por token limita peticiones por minuto (HTTP 429): se reintenta con espera creciente."""
    for k in range(intentos):
        try:
            return cliente.predict(endpoint=endpoint, inputs={"input": entradas})
        except Exception as e:
            if "429" not in str(e) or k == intentos - 1:
                raise
            time.sleep(min(2 ** k, 60))

def embeber(textos, endpoint=EMB_ENDPOINT, lote=16):
    cliente = get_deploy_client("databricks")
    vecs = []
    for i in range(0, len(textos), lote):
        r = _predict_con_reintentos(cliente, endpoint, [t or "-" for t in textos[i:i + lote]])
        vecs += [d["embedding"] for d in r["data"]]
    v = np.array(vecs, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def ajustar_embeddings(endpoint=EMB_ENDPOINT):
    return {f: embeber(list(galeria[f"str_{f}"]), endpoint) for f in FACETAS if (galeria[f"str_{f}"] != "").any()}

SIMS_EMB = {}   # caché: endpoint → {faceta: matriz coseno consulta × ficha}; el ajuste fino no vuelve a llamar al endpoint

def matriz_embeddings(emb_galeria, endpoint=EMB_ENDPOINT, pesos=PESOS):
    if endpoint not in SIMS_EMB:
        sims = {}
        for f, eg in emb_galeria.items():
            q, g = list(consultas[f"str_{f}"]), list(galeria[f"str_{f}"])
            sims[f] = _sin_vacios(embeber(q, endpoint) @ eg.T, q, g)
        SIMS_EMB[endpoint] = sims
    return sum((pesos[f] * s for f, s in SIMS_EMB[endpoint].items()), np.zeros((len(consultas), len(galeria))))

# --- 6 · Cross-encoder (preentrenado, zero-shot) --------------------------------------
from sentence_transformers import CrossEncoder

def ajustar_cross_encoder(modelo=CROSS_ENCODER):
    return CrossEncoder(modelo, max_length=256)

SIMS_CE = {}    # caché: faceta → matriz de logits del cross-encoder (NaN = faceta ausente en algún lado)

def matriz_cross_encoder(ce, pesos=PESOS, lote=64, transformacion="logit"):
    if not SIMS_CE:
        for f in FACETAS:
            q, g = list(consultas[f"str_{f}"]), list(galeria[f"str_{f}"])
            s = np.full((len(consultas), len(galeria)), np.nan)
            idx = [(i, j) for i, j in zip(cq, cg) if q[i] and g[j]]      # solo pares con la faceta en ambos lados
            if idx:
                v = ce.predict([(q[i], g[j]) for i, j in idx], batch_size=lote, show_progress_bar=False)
                s[tuple(np.array(idx).T)] = v
            SIMS_CE[f] = s
    m = np.zeros((len(consultas), len(galeria)))
    for f, s in SIMS_CE.items():
        t = 1 / (1 + np.exp(-s)) if transformacion == "sigmoide" else s
        m += pesos[f] * np.nan_to_num(t, nan=0.0)
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
# MAGIC - **Qué se ajusta:** los dos modelos con mayor cobertura en la sección 6 (sin contar el azar). Cada modelo tiene su
# MAGIC   espacio de búsqueda; el peso de la faceta *vestimenta* se fija en 1 y se ajusta el de *marcas*, porque la métrica
# MAGIC   es invariante a la escala y solo importa la proporción entre facetas.
# MAGIC - **Cómo:** Optuna (TPE, semilla fija), `n_trials` intentos por modelo (50 por defecto), cada intento es un run hijo
# MAGIC   bajo el run padre `ajuste_top2`.
# MAGIC - **Generalización sin etiquetas:** las necropsias se parten en dos mitades fijas por hash del identificador.
# MAGIC   Optuna maximiza la cobertura en la mitad **A (ajuste)**; la mitad **B (validación)** no interviene en la búsqueda y
# MAGIC   sirve para comprobar que la mejora no es sobreajuste a esas consultas.
# MAGIC - El umbral de la métrica (z ≥ 5) **no** se ajusta: es parte de la definición de la métrica, no del modelo.
# MAGIC
# MAGIC | Modelo | Hiperparámetros |
# MAGIC |---|---|
# MAGIC | Reglas SQL | Peso de cada regla (prenda + color, prenda, marca, región de tatuaje) y peso de *marcas* |
# MAGIC | Fuzzy | Función de similitud (`token_set`, `token_sort`, `partial_token_set`, `WRatio`) y peso de *marcas* |
# MAGIC | TF-IDF | n-gramas (1 o 1–2), `sublinear_tf`, `min_df` y peso de *marcas* |
# MAGIC | BM25 | Variante (Okapi, L, Plus), `k1`, `b` y peso de *marcas* |
# MAGIC | Splink | Niveles de intersección de conjuntos, iteraciones de EM y probabilidad previa de coincidencia |
# MAGIC | Embeddings | Modelo (qwen3 multilingüe, gte-large, bge-large) y peso de *marcas* |
# MAGIC | Cross-encoder | Transformación del score (logit o sigmoide) y peso de *marcas* |

# COMMAND ----------

dbutils.widgets.text("n_trials", "50", "Intentos de Optuna por modelo")
N_TRIALS = int(dbutils.widgets.get("n_trials"))

import hashlib
import optuna
from pyspark.sql.window import Window
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
optuna.logging.set_verbosity(optuna.logging.WARNING)

mitad = consultas["id_consulta"].map(lambda x: int(hashlib.sha256(x.encode()).hexdigest(), 16) % 2)
CONS_A = set(consultas.loc[mitad == 0, "id_consulta"])
CONS_B = set(consultas.loc[mitad == 1, "id_consulta"])
print(f"Consultas para ajuste (A): {len(CONS_A)} | para validación (B): {len(CONS_B)}")


def _pesos(p):
    return {"vest": 1.0, "marcas": p.get("peso_marcas", 1.0)}


def _medir(entrenar, inferir):
    t0 = time.perf_counter()
    obj = entrenar()
    t1 = time.perf_counter()
    rk = inferir(obj)
    return rk, t1 - t0, time.perf_counter() - t1


# Para cada modelo: (espacio de búsqueda: trial → params, puntuación: params → (ranking, t_entrenamiento, t_inferencia))
ESPACIOS = {
    "0_sql_reglas": (
        lambda t: {"prenda_color": t.suggest_float("prenda_color", 0, 4), "prenda": t.suggest_float("prenda", 0, 3),
                   "marca": t.suggest_float("marca", 0, 4), "region_marca": t.suggest_float("region_marca", 0, 4),
                   "peso_marcas": t.suggest_float("peso_marcas", 0, 3)},
        lambda p: _medir(lambda: None, lambda _: ranking_sql({k: p[k] for k in PESOS_REGLAS}, _pesos(p)))),
    "1_fuzzy": (
        lambda t: {"scorer": t.suggest_categorical("scorer", list(SCORERS)),
                   "peso_marcas": t.suggest_float("peso_marcas", 0, 3)},
        lambda p: _medir(lambda: None, lambda _: ranking_desde_matriz(matriz_fuzzy(p["scorer"], _pesos(p))))),
    "2_tfidf": (
        lambda t: {"ngramas": t.suggest_categorical("ngramas", [1, 2]),
                   "sublinear_tf": t.suggest_categorical("sublinear_tf", [True, False]),
                   "min_df": t.suggest_int("min_df", 1, 3), "peso_marcas": t.suggest_float("peso_marcas", 0, 3)},
        lambda p: _medir(lambda: ajustar_tfidf(p["ngramas"], p["sublinear_tf"], p["min_df"]),
                         lambda v: ranking_desde_matriz(matriz_tfidf(v, p["ngramas"], _pesos(p))))),
    "3_bm25": (
        lambda t: {"variante": t.suggest_categorical("variante", list(VARIANTES_BM25)),
                   "k1": t.suggest_float("k1", 0.3, 3.0), "b": t.suggest_float("b", 0.0, 1.0),
                   "peso_marcas": t.suggest_float("peso_marcas", 0, 3)},
        lambda p: _medir(lambda: ajustar_bm25(p["k1"], p["b"], p["variante"]),
                         lambda idx: ranking_desde_matriz(matriz_bm25(idx, _pesos(p))))),
    "4_splink_em": (
        lambda t: {"niveles": t.suggest_categorical("niveles", ["1", "2,1", "3,2,1"]),
                   "em_iter": t.suggest_categorical("em_iter", [10, 25, 50]),
                   "factor_previa": t.suggest_float("factor_previa", 0.1, 10.0, log=True)},
        lambda p: _medir(lambda: ajustar_splink(tuple(int(x) for x in p["niveles"].split(",")),
                                                em_iter=p["em_iter"], factor_previa=p["factor_previa"]),
                         ranking_splink)),
    "5_embeddings": (
        lambda t: {"endpoint": t.suggest_categorical("endpoint", [EMB_ENDPOINT, "databricks-gte-large-en",
                                                                  "databricks-bge-large-en"]),
                   "peso_marcas": t.suggest_float("peso_marcas", 0, 3)},
        lambda p: _medir(lambda: None if p["endpoint"] in SIMS_EMB else ajustar_embeddings(p["endpoint"]),
                         lambda eg: ranking_desde_matriz(matriz_embeddings(eg, p["endpoint"], _pesos(p))))),
    "6_cross_encoder": (
        lambda t: {"transformacion": t.suggest_categorical("transformacion", ["logit", "sigmoide"]),
                   "peso_marcas": t.suggest_float("peso_marcas", 0, 3)},
        lambda p: _medir(lambda: None, lambda _: ranking_desde_matriz(
            matriz_cross_encoder(ce, _pesos(p), transformacion=p["transformacion"])))),
}


def evaluar_mitades(rk):
    mA, _ = evaluar(rk[rk["id_consulta"].isin(CONS_A)])
    mB, _ = evaluar(rk[rk["id_consulta"].isin(CONS_B)])
    return mA, mB


TOP2 = list(tabla.loc[tabla["modelo"] != "B0_azar"].sort_values("cobertura", ascending=False)["modelo"].head(2))
FAMILIA = dict(zip(tabla["modelo"], tabla["familia"]))
print("Modelos a ajustar:", TOP2)

# COMMAND ----------

resumen_ajuste, estudios, runs_ajustados = [], {}, {}

with fase_padre("ajuste_top2", f"Optuna sobre {TOP2}: {N_TRIALS} intentos por modelo; objetivo = cobertura en la mitad A"):
    mlflow.log_params({"n_trials": N_TRIALS, "modelos": ",".join(TOP2), "sampler": "TPE", "objetivo": "cobertura_mitad_A"})

    for nombre in TOP2:
        espacio, puntuar = ESPACIOS[nombre]

        def objetivo(trial, nombre=nombre, espacio=espacio, puntuar=puntuar):
            p = espacio(trial)
            with corrida(f"{nombre}__intento_{trial.number:03d}", FAMILIA[nombre], "ajuste_top2", p,
                         tags={"modelo_base": nombre, "intento": str(trial.number)}):
                rk, te, ti = puntuar(p)
                mA, mB = evaluar_mitades(rk)
                mlflow.log_metrics({"cobertura": mA["cobertura"], "margen_mediano": mA["margen_mediano"],
                                    "cobertura_validacion": mB["cobertura"], "margen_validacion": mB["margen_mediano"],
                                    "tiempo_entrenamiento_s": te, "tiempo_inferencia_s": ti})
            trial.set_user_attr("cobertura_validacion", mB["cobertura"])
            return mA["cobertura"]

        estudio = optuna.create_study(direction="maximize", study_name=nombre,
                                      sampler=optuna.samplers.TPESampler(seed=SEMILLA))
        estudio.optimize(objetivo, n_trials=N_TRIALS)
        estudios[nombre] = estudio

        fig_h = plot_optimization_history(estudio).figure
        fig_h.suptitle(f"Historia de optimización — {nombre}")
        mlflow.log_figure(fig_h, f"graficas/{nombre}_historia.png")
        try:
            fig_i = plot_param_importances(estudio).figure
            fig_i.suptitle(f"Importancia de hiperparámetros — {nombre}")
            mlflow.log_figure(fig_i, f"graficas/{nombre}_importancia.png")
        except Exception as e:
            print(f"{nombre}: sin gráfica de importancia ({e})")
        plt.close("all")

        # Modelo ajustado: mejores hiperparámetros, evaluado en todas las consultas y en cada mitad
        with corrida(f"{nombre}_ajustado", FAMILIA[nombre], "ajuste_top2", estudio.best_params,
                     tags={"modelo_base": nombre, "ajustado": "true"}) as c:
            rk, te, ti = puntuar(estudio.best_params)
            c.tiempos = {"tiempo_entrenamiento_s": te, "tiempo_inferencia_s": ti}
            m = c.registrar(rk)
            mA, mB = evaluar_mitades(rk)
            mlflow.log_metrics({"cobertura_ajuste": mA["cobertura"], "cobertura_validacion": mB["cobertura"],
                                "cobertura_validacion_ic_inf": mB["cobertura_ic_inf"],
                                "cobertura_validacion_ic_sup": mB["cobertura_ic_sup"]})
            runs_ajustados[nombre] = c.run.info.run_id

        base = (spark.table(T_RANKINGS).where(F.col("run_id") == resultados[nombre])
                     .select("id_consulta", "id_ficha", "score").toPandas())
        bA, bB = evaluar_mitades(base)
        resumen_ajuste.append({
            "modelo": nombre, "familia": FAMILIA[nombre], "intentos": len(estudio.trials),
            "cobertura_base_validacion": bB["cobertura"],
            "cobertura_ajustada_ajuste": mA["cobertura"], "cobertura_ajustada_validacion": mB["cobertura"],
            "validacion_ic_inf": mB["cobertura_ic_inf"], "validacion_ic_sup": mB["cobertura_ic_sup"],
            "cobertura_ajustada_todas": m["cobertura"], "margen_mediano": m["margen_mediano"],
            "tiempo_entrenamiento_s": te, "tiempo_inferencia_ms_por_consulta": 1000 * ti / max(len(consultas), 1),
            "mejores_hiperparametros": json.dumps(estudio.best_params, default=str),
        })

    resumen_ajuste = pd.DataFrame(resumen_ajuste)
    mlflow.log_table(resumen_ajuste, "resumen_ajuste.json")      # solo métricas agregadas e hiperparámetros

display(resumen_ajuste)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Elección del modelo final
# MAGIC
# MAGIC **Criterios, en orden:**
# MAGIC 1. **Cobertura en validación (mitad B)** del modelo ajustado: la métrica principal medida en consultas que no
# MAGIC    intervinieron en el ajuste.
# MAGIC 2. Si los intervalos de confianza de los dos modelos se traslapan, la diferencia no es concluyente y se prefiere el
# MAGIC    **más interpretable para el perito**: que pueda explicar por qué propone un candidato (qué prenda, color o
# MAGIC    tatuaje coincide). Orden de interpretabilidad: reglas SQL > BM25 > TF-IDF > fuzzy > Splink > embeddings >
# MAGIC    cross-encoder.
# MAGIC 3. Como contexto: margen (qué tan separado queda el mejor candidato), concordancia con los demás métodos y tiempos.
# MAGIC
# MAGIC El modelo elegido genera la tabla `candidatos_top10_modelo_final` en Unity Catalog: las 10 fichas candidatas de cada
# MAGIC necropsia no identificada, para revisión pericial. Esa tabla no se imprime ni se registra en MLflow.

# COMMAND ----------

INTERPRETABILIDAD = ["0_sql_reglas", "3_bm25", "2_tfidf", "1_fuzzy", "4_splink_em", "5_embeddings", "6_cross_encoder"]

r = resumen_ajuste.set_index("modelo")
a, b = sorted(TOP2, key=lambda x: -r.loc[x, "cobertura_ajustada_validacion"])
traslape = r.loc[b, "validacion_ic_sup"] >= r.loc[a, "validacion_ic_inf"]
if traslape:
    FINAL = min(TOP2, key=INTERPRETABILIDAD.index)
    CRITERIO = (f"IC 95 % de validación traslapados ({a}: {r.loc[a, 'validacion_ic_inf']:.3f}–{r.loc[a, 'validacion_ic_sup']:.3f}; "
                f"{b}: {r.loc[b, 'validacion_ic_inf']:.3f}–{r.loc[b, 'validacion_ic_sup']:.3f}) → se elige el más interpretable")
else:
    FINAL = a
    CRITERIO = f"{a} supera a {b} en cobertura de validación con IC 95 % sin traslape"

conc = concordancia_top10(list(runs_ajustados.values()))
print(f"Modelo final: {FINAL}\nCriterio: {CRITERIO}")
print("Concordancia top-10 entre los dos ajustados:",
      {nombre: round(conc.get(rid, float("nan")), 3) for nombre, rid in runs_ajustados.items()})

top10 = (spark.table(T_RANKINGS).where(F.col("run_id") == runs_ajustados[FINAL])
              .join(spark.table(T_NECROPSIAS).select(F.col("no_nec").cast("string").alias("id_consulta"), "status"),
                    "id_consulta")
              .where(F.upper("status").like("NO IDENT%"))
              .withColumn("rango", F.row_number().over(Window.partitionBy("id_consulta").orderBy(F.desc("score"))))
              .where("rango <= 10")
              .select("id_consulta", "id_ficha", "score", "rango", F.lit(FINAL).alias("modelo")))
top10.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(_t("candidatos_top10_modelo_final"))

with fase_padre("final", f"Modelo final: {FINAL}. {CRITERIO}"):
    mlflow.log_params({"modelo_final": FINAL, "criterio": CRITERIO[:480], "finalistas": ",".join(TOP2),
                       "run_modelo_final": runs_ajustados[FINAL]})
    mlflow.log_metrics({"cobertura_validacion_final": float(r.loc[FINAL, "cobertura_ajustada_validacion"]),
                        "necropsias_no_identificadas_con_top10": top10.select("id_consulta").distinct().count()})
    mlflow.log_table(resumen_ajuste, "resumen_finalistas.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Justificación del modelo final
# MAGIC
# MAGIC *Resultados de la ejecución del 25 de octubre de 2026 (galería `facetas`, 884 necropsias, 50 intentos por modelo).*
# MAGIC
# MAGIC | Finalista | Cobertura base | Cobertura ajustada (A, ajuste) | Cobertura ajustada (B, validación) | Margen | Inferencia por consulta |
# MAGIC |---|---|---|---|---|---|
# MAGIC | **BM25** (variante L, k1 ≈ 3.0, b ≈ 0.01, peso marcas ≈ 0.12) | 0.469 | 0.692 | **0.699** (IC 95 % 0.655–0.745) | 1.98 | 0.52 ms |
# MAGIC | TF-IDF (1–2-gramas, sin sublinear, peso marcas ≈ 3.0) | 0.602 | 0.613 | 0.609 (IC 95 % 0.566–0.657) | 0.83 | 0.03 ms |
# MAGIC
# MAGIC **Modelo final: BM25 ajustado.**
# MAGIC 1. **Métrica principal.** Es el de mayor cobertura en la mitad de validación (0.699 frente a 0.609). Los intervalos
# MAGIC    apenas se tocan (0.655 frente a 0.657), así que la diferencia no se declara concluyente. El desempate por
# MAGIC    interpretabilidad apunta al mismo modelo.
# MAGIC 2. **Generalización.** La cobertura en la mitad B (0.699), que no intervino en la búsqueda, es igual a la de la mitad A
# MAGIC    (0.692): la mejora no es sobreajuste a las consultas usadas para ajustar.
# MAGIC 3. **El ajuste sí aportó.** BM25 pasó de 0.469 a 0.699 en validación (+0.23); TF-IDF casi no cambió (0.602 → 0.609).
# MAGIC 4. **Interpretabilidad para el perito.** El score de BM25 es una suma de contribuciones por palabra coincidente,
# MAGIC    ponderadas por su rareza en las fichas: se puede mostrar al perito qué prendas, colores o marcas explican cada
# MAGIC    candidato. Es más legible que un coseno de TF-IDF y mucho más que embeddings o el cross-encoder.
# MAGIC 5. **Costo.** Sin entrenamiento y con 0.5 ms por consulta, puede correr sobre todo el libro en segundos.
# MAGIC
# MAGIC **Lectura de los hiperparámetros.** `b ≈ 0` apaga la normalización por longitud: una ficha con descripción larga no
# MAGIC se penaliza, lo cual tiene sentido porque más prendas descritas es más evidencia, no ruido. `k1 ≈ 3` hace que repetir
# MAGIC un término siga sumando. El peso de *marcas* cae a ≈ 0.12 porque casi ninguna necropsia registra tatuajes (12 de
# MAGIC 1,246), así que esa faceta aporta poco a la cobertura.
# MAGIC
# MAGIC **Limitaciones (declaradas).**
# MAGIC - La cobertura mide que el modelo propone un candidato que **destaca**, no que sea el correcto. Optimizarla favorece
# MAGIC   scores más "picudos"; la validación en la mitad B descarta el sobreajuste a consultas, pero no confirma aciertos.
# MAGIC   La confirmación es pericial: la tabla `candidatos_top10_modelo_final` deja 620 necropsias no identificadas con sus
# MAGIC   10 fichas candidatas para esa revisión.
# MAGIC - `k1` y el peso de *marcas* de BM25, y el peso de *marcas* de TF-IDF, quedaron en el borde de su rango de búsqueda;
# MAGIC   ampliar esos rangos es el siguiente paso del ajuste.
# MAGIC - Bajar el peso de los tatuajes es lo que maximiza la cobertura con los datos actuales, pero en un caso concreto un
# MAGIC   tatuaje coincidente es evidencia muy fuerte. Se recomienda que el SEMEFO registre tatuajes de forma sistemática y
# MAGIC   que la salida al perito siempre muestre las coincidencias de tatuajes aunque pesen poco en el score.
