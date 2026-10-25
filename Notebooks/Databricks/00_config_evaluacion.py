# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Configuración común y arnés de evaluación (Avance 4)
# MAGIC
# MAGIC **Proyecto Integrador MNA — Cotejo ante-mortem / post-mortem (Querétaro)**
# MAGIC
# MAGIC Todos los notebooks de modelos cargan este con `%run ./00_config_evaluacion`. Aquí vive lo que tiene que ser
# MAGIC **idéntico** para que la comparación sea justa:
# MAGIC
# MAGIC | Pieza | Qué garantiza |
# MAGIC |---|---|
# MAGIC | Nombres de tablas y parámetros | Todos leen las mismas entradas y escriben en el mismo lugar |
# MAGIC | Contrato de ranking `(id_consulta, id_ficha, score)` | Cada modelo entrega lo mismo; nadie calcula su propia métrica |
# MAGIC | `evaluar()` | Cobertura (métrica principal), margen top-1 vs top-2, IC 95 % por bootstrap — **sin etiquetas** |
# MAGIC | `corrida()` | Run de MLflow con tags, parámetros, versiones Delta de las entradas y **tiempos de entrenamiento e inferencia** |
# MAGIC | `tabla_comparativa()` | Modelos ordenados por cobertura, con margen, concordancia entre métodos y tiempos |
# MAGIC
# MAGIC **Cotejo no supervisado (Decisión 24):** no hay pares ficha ↔ necropsia confirmados, así que no hay Recall@k ni MRR.
# MAGIC **Dirección de la consulta (Decisión 20):** cada consulta es un registro de necropsia; la galería son las fichas.
# MAGIC
# MAGIC **Reglas de privacidad en MLflow**
# MAGIC - Solo parámetros, métricas agregadas y gráficas de agregados. Nunca filas, textos ni identificadores.
# MAGIC - `autolog` sin datasets ni ejemplos de entrada; tracing de LLM desactivado.
# MAGIC - Los rankings se guardan en Unity Catalog (`resultados_rankings`, con `run_id`), no como artefacto de MLflow.

# COMMAND ----------

# MAGIC %md ## 1. Parámetros

# COMMAND ----------

dbutils.widgets.text("catalog", "proyecto_mna", "Catálogo")
dbutils.widgets.text("schema", "fichas_busqueda", "Esquema")
dbutils.widgets.text("experimento", "/Shared/proyecto_mna/avance4_cotejo", "Experimento MLflow")
dbutils.widgets.text("semilla", "23", "Semilla")
dbutils.widgets.text("n_bootstrap", "1000", "Réplicas bootstrap para IC 95 %")
dbutils.widgets.text("z_umbral", "5", "z robusto mínimo para contar una consulta como cubierta")
dbutils.widgets.dropdown("autoprueba", "false", ["true", "false"], "Correr autoprueba con scores simulados")

# COMMAND ----------

import time
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from pyspark.sql import functions as F

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
EXPERIMENTO = dbutils.widgets.get("experimento")
SEMILLA = int(dbutils.widgets.get("semilla"))
N_BOOTSTRAP = int(dbutils.widgets.get("n_bootstrap"))
Z_UMBRAL = float(dbutils.widgets.get("z_umbral"))
AUTOPRUEBA = dbutils.widgets.get("autoprueba") == "true"

METRICA_PRINCIPAL = "cobertura"
METRICAS_SECUNDARIAS = ["margen_mediano", "cobertura_z2", "concordancia_top10"]
MIN_CANDIDATOS = 15     # con menos candidatos la escala robusta es inestable y el ruido "destaca" seguido
TOPE_Z = 10.0           # z infinito (resto empatado) se acota para promediar márgenes

def _t(nombre):
    return f"{CATALOG}.{SCHEMA}.{nombre}"

# Entradas (ya existen; no se regeneran)
T_NECROPSIAS = _t("necropsias_enriquecidas")       # consultas
T_FICHAS_GOLD = _t("fichas_vector_search_gold")     # galería: 75 fichas, texto_canonico_fe + filtros
T_FICHAS_TEXTO = _t("fichas_texto_embedding")      # galería: 60 fichas con texto por faceta (vestimenta, tatuajes, señas)
T_FICHAS_VESTIMENTA = _t("fichas_vestimenta_silver")
T_FICHAS_TATUAJES = _t("fichas_tatuajes_silver")

# Salidas del Avance 4
T_CANDIDATOS = _t("candidatos_bloqueados")          # pares que sobreviven el blocking
T_RANKINGS = _t("resultados_rankings")              # rankings de cada run, con run_id

ENTRADAS_VERSIONADAS = [T_NECROPSIAS, T_FICHAS_GOLD, T_FICHAS_TEXTO, T_CANDIDATOS]

# COMMAND ----------

# MAGIC %md ## 2. MLflow: experimento y salvaguardas de privacidad

# COMMAND ----------

import posixpath
from databricks.sdk import WorkspaceClient

mlflow.set_registry_uri("databricks-uc")
WorkspaceClient().workspace.mkdirs(posixpath.dirname(EXPERIMENTO))   # set_experiment no crea carpetas
mlflow.set_experiment(EXPERIMENTO)

# Ningún autolog global: cada notebook lo activa explícitamente, y siempre sin datos.
mlflow.autolog(disable=True)
AUTOLOG_SEGURO = dict(log_datasets=False, log_input_examples=False, log_model_signatures=False,
                      silent=True)

# El tracing de LLM guardaría prompts con texto real de fichas y necropsias.
try:
    mlflow.tracing.disable()
except Exception:
    pass

print("Experimento:", EXPERIMENTO)

# COMMAND ----------

# MAGIC %md ## 3. Versiones Delta de las entradas
# MAGIC
# MAGIC Se registra el número de versión de cada tabla, no su contenido: basta para reproducir una corrida con *time travel*.

# COMMAND ----------

def versiones_tablas(tablas=ENTRADAS_VERSIONADAS):
    versiones = {}
    for t in tablas:
        try:
            v = spark.sql(f"DESCRIBE HISTORY {t} LIMIT 1").first()["version"]
            versiones[f"version__{t.split('.')[-1]}"] = int(v)
        except Exception:
            versiones[f"version__{t.split('.')[-1]}"] = -1   # aún no existe
    return versiones

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluación sin etiquetas
# MAGIC
# MAGIC Cada método puntúa en su propia escala (BM25 no tiene tope, el coseno va de 0 a 1, las reglas cuentan
# MAGIC coincidencias), así que un umbral fijo de score no sirve para compararlos. Por eso todo se mide **dentro de cada
# MAGIC consulta**, contra sus propios candidatos:
# MAGIC
# MAGIC - **z del mejor candidato** = (score del 1.º − mediana del resto) / escala del resto, con escala = 1.4826 × MAD
# MAGIC   (si es 0, la desviación estándar; si también es 0, z = ∞ cuando el 1.º supera al resto y 0 si empata).
# MAGIC - **Cobertura (métrica principal)** = % de consultas evaluables cuyo mejor candidato tiene z ≥ `z_umbral` (5):
# MAGIC   el método propone al perito un candidato que destaca claramente.
# MAGIC - **Calibración del umbral** (simulación con scores normales puro ruido): con z ≥ 3 un método aleatorio "cubre" 16–19 %
# MAGIC   de las consultas; con z ≥ 5 y ≥ 15 candidatos, ≤ 2.5 % (0.1 % con 70 candidatos). Por eso z = 5 y mínimo 15.
# MAGIC - **Margen mediano** = mediana de (1.º − 2.º) / escala: qué tan separado queda el mejor del segundo.
# MAGIC - Consultas con menos de 15 candidatos no son evaluables y se reportan aparte.
# MAGIC
# MAGIC **Limitación:** la cobertura mide que el método encuentra una señal fuerte, **no** que el candidato sea correcto.
# MAGIC Eso solo lo puede confirmar el perito.

# COMMAND ----------

def senal_por_consulta(rankings: pd.DataFrame) -> pd.DataFrame:
    """Una fila por consulta: n de candidatos, z robusto del mejor candidato y margen sobre el segundo."""
    filas = []
    for q, g in rankings.groupby("id_consulta", sort=False):
        s = np.sort(g["score"].to_numpy(dtype=float))[::-1]
        n = len(s)
        if n < MIN_CANDIDATOS:
            filas.append((q, n, np.nan, np.nan))
            continue
        resto = s[1:]
        med = np.median(resto)
        escala = 1.4826 * np.median(np.abs(resto - med))
        if escala == 0:
            escala = resto.std()
        if escala == 0:
            z = np.inf if s[0] > med else 0.0
            margen = np.inf if s[0] > s[1] else 0.0
        else:
            z = (s[0] - med) / escala
            margen = (s[0] - s[1]) / escala
        filas.append((q, n, z, margen))
    return pd.DataFrame(filas, columns=["id_consulta", "n_candidatos", "z_top1", "margen"])


def _ic_bootstrap(indicador: np.ndarray, rng) -> tuple:
    n = len(indicador)
    reps = [indicador[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def evaluar(rankings: pd.DataFrame) -> tuple:
    """Métricas agregadas sin etiquetas de un ranking."""
    rng = np.random.default_rng(SEMILLA)
    sp = senal_por_consulta(rankings)
    ev = sp.dropna(subset=["z_top1"])
    cubierta = (ev["z_top1"] >= Z_UMBRAL).to_numpy(dtype=float)

    m = {"cobertura": float(cubierta.mean()) if len(ev) else 0.0}
    m["cobertura_ic_inf"], m["cobertura_ic_sup"] = _ic_bootstrap(cubierta, rng) if len(ev) else (0.0, 0.0)
    m["cobertura_z2"] = float((ev["z_top1"] >= 2).mean()) if len(ev) else 0.0
    m["margen_mediano"] = float(np.median(np.minimum(ev["margen"], TOPE_Z))) if len(ev) else 0.0
    m["consultas_total"] = int(len(sp))
    m["consultas_evaluables"] = int(len(ev))
    m["consultas_pocos_candidatos"] = int(len(sp) - len(ev))
    m["candidatos_por_consulta"] = float(sp["n_candidatos"].mean()) if len(sp) else 0.0
    return m, sp

# COMMAND ----------

# MAGIC %md ## 5. Gráficas (solo agregados)

# COMMAND ----------

def graficas_agregadas(sp: pd.DataFrame, titulo: str):
    ev = sp.dropna(subset=["z_top1"])
    z = np.minimum(ev["z_top1"].to_numpy(dtype=float), TOPE_Z)
    umbrales = np.linspace(0, TOPE_Z, 101)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(umbrales, [(z >= u).mean() if len(z) else 0 for u in umbrales], color="#2b6cb0")
    ax[0].axvline(Z_UMBRAL, color="#c53030", ls="--", lw=1, label=f"umbral z = {Z_UMBRAL:g}")
    ax[0].set(title="Cobertura según umbral", xlabel="z robusto del mejor candidato", ylabel="cobertura",
              ylim=(0, 1.02))
    ax[0].legend(loc="upper right")

    ax[1].hist(z, bins=np.linspace(0, TOPE_Z, 41), color="#2b6cb0")
    ax[1].axvline(Z_UMBRAL, color="#c53030", ls="--", lw=1)
    ax[1].set(title=f"z del mejor candidato (acotado a {TOPE_Z:g})", xlabel="z", ylabel="consultas")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Corrida de MLflow
# MAGIC
# MAGIC Uso desde un notebook de modelo:
# MAGIC ```python
# MAGIC with fase_padre("comparacion"):
# MAGIC     with corrida("3_bm25", familia="lexico", fase="comparacion", params={"k1": 1.2, "b": 0.75}) as c:
# MAGIC         with c.entrenamiento():
# MAGIC             indice = construir_bm25(fichas)
# MAGIC         with c.inferencia():
# MAGIC             rankings = puntuar(indice, candidatos)     # pandas: id_consulta, id_ficha, score
# MAGIC         c.registrar(rankings)
# MAGIC ```
# MAGIC Ningún método se entrena con etiquetas. "Entrenamiento" es lo que el método construye antes de consultar:
# MAGIC índice BM25 o TF-IDF, estimación EM de Splink, embeddings de la galería. Si no construye nada, el tiempo es 0.

# COMMAND ----------

class _Corrida:
    def __init__(self, run, nombre):
        self.run = run
        self.nombre = nombre
        self.tiempos = {"tiempo_entrenamiento_s": 0.0, "tiempo_inferencia_s": 0.0}

    @contextlib.contextmanager
    def _cronometro(self, clave):
        t0 = time.perf_counter()
        yield
        self.tiempos[clave] += time.perf_counter() - t0

    def entrenamiento(self):
        return self._cronometro("tiempo_entrenamiento_s")

    def inferencia(self):
        return self._cronometro("tiempo_inferencia_s")

    def registrar(self, rankings: pd.DataFrame, guardar_rankings=True):
        faltan = {"id_consulta", "id_ficha", "score"} - set(rankings.columns)
        assert not faltan, f"El ranking no cumple el contrato; faltan columnas: {faltan}"
        rankings = rankings[["id_consulta", "id_ficha", "score"]]

        metricas, sp = evaluar(rankings)
        n = max(metricas["consultas_total"], 1)
        metricas.update(self.tiempos)
        metricas["tiempo_inferencia_ms_por_consulta"] = 1000 * self.tiempos["tiempo_inferencia_s"] / n
        mlflow.log_metrics(metricas)
        mlflow.log_figure(graficas_agregadas(sp, self.nombre), "graficas/cobertura_y_z.png")
        plt.close("all")

        if guardar_rankings:
            (spark.createDataFrame(rankings)
                  .selectExpr("CAST(id_consulta AS STRING) id_consulta", "CAST(id_ficha AS STRING) id_ficha",
                              "CAST(score AS DOUBLE) score")
                  .withColumn("run_id", F.lit(self.run.info.run_id))
                  .withColumn("modelo", F.lit(self.nombre))
                  .write.mode("append").saveAsTable(T_RANKINGS))

        print(f"{self.nombre}: cobertura = {metricas['cobertura']:.3f} "
              f"[{metricas['cobertura_ic_inf']:.3f}–{metricas['cobertura_ic_sup']:.3f}], "
              f"margen mediano = {metricas['margen_mediano']:.2f}, "
              f"entrenamiento = {self.tiempos['tiempo_entrenamiento_s']:.1f} s")
        return metricas


@contextlib.contextmanager
def corrida(nombre, familia, fase, params=None, tags=None):
    """Abre un run (anidado bajo el run activo, normalmente el de fase_padre) con tags y parámetros comunes."""
    with mlflow.start_run(run_name=nombre, nested=mlflow.active_run() is not None) as run:
        mlflow.set_tags({"modelo": nombre, "familia": familia, "fase": fase, "supervision": "ninguna",
                         "direccion": "necropsia_a_fichas", "es_padre": "false", **(tags or {})})
        mlflow.log_params({"semilla": SEMILLA, "metrica_principal": METRICA_PRINCIPAL, "z_umbral": Z_UMBRAL,
                           "min_candidatos": MIN_CANDIDATOS, **(params or {})})
        mlflow.log_params(versiones_tablas())
        yield _Corrida(run, nombre)


@contextlib.contextmanager
def fase_padre(fase, descripcion=""):
    """Run padre que agrupa una fase: 'comparacion', 'ajuste_top2' o 'final'."""
    with mlflow.start_run(run_name=fase) as run:
        mlflow.set_tags({"fase": fase, "es_padre": "true"})
        if descripcion:
            mlflow.set_tag("mlflow.note.content", descripcion)
        yield run

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Comparativa
# MAGIC
# MAGIC Tabla para la rúbrica: modelos **ordenados por cobertura**, con métricas secundarias y tiempos.
# MAGIC La **concordancia top-10** es el Jaccard promedio entre el top-10 de un método y el de cada uno de los demás:
# MAGIC indica si encuentra candidatos que otros métodos también ven, o si se aparta de todos.

# COMMAND ----------

def concordancia_top10(run_ids: list) -> dict:
    r = (spark.table(T_RANKINGS).where(F.col("run_id").isin(run_ids))
              .select("run_id", "id_consulta", "id_ficha", "score").toPandas())
    r = r.sort_values(["run_id", "id_consulta", "score"], ascending=[True, True, False])
    top = r.groupby(["run_id", "id_consulta"]).head(10).groupby(["run_id", "id_consulta"])["id_ficha"].apply(frozenset)
    tops = {rid: top.xs(rid, level="run_id") for rid in run_ids if rid in top.index.get_level_values(0)}
    out = {}
    for a in tops:
        valores = []
        for b in tops:
            if a == b:
                continue
            comunes = tops[a].index.intersection(tops[b].index)
            valores += [len(tops[a][q] & tops[b][q]) / len(tops[a][q] | tops[b][q]) for q in comunes]
        out[a] = float(np.mean(valores)) if valores else np.nan
    return out


def tabla_comparativa(fase="comparacion", filtro=""):
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENTO],
        filter_string=f"tags.fase = '{fase}' AND tags.es_padre = 'false'" + (f" AND {filtro}" if filtro else ""),
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        print("Sin corridas para", fase)
        return runs
    runs = runs.drop_duplicates("tags.modelo")          # la corrida más reciente de cada modelo
    runs["metrics.concordancia_top10"] = runs["run_id"].map(concordancia_top10(list(runs["run_id"])))
    cols = (["tags.modelo", "tags.familia", f"metrics.{METRICA_PRINCIPAL}",
             f"metrics.{METRICA_PRINCIPAL}_ic_inf", f"metrics.{METRICA_PRINCIPAL}_ic_sup"]
            + [f"metrics.{m}" for m in METRICAS_SECUNDARIAS]
            + ["metrics.consultas_evaluables", "metrics.tiempo_entrenamiento_s",
               "metrics.tiempo_inferencia_ms_por_consulta", "run_id"])
    cols = [c for c in cols if c in runs.columns]
    return (runs[cols].sort_values(f"metrics.{METRICA_PRINCIPAL}", ascending=False)
                      .rename(columns=lambda c: c.split(".", 1)[-1]).reset_index(drop=True))


def grafica_comparativa(t: pd.DataFrame, titulo="Comparativa de métodos (sin etiquetas)"):
    t = t.iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.5 * len(t) + 1.5), sharey=True)
    err = [t["cobertura"] - t["cobertura_ic_inf"], t["cobertura_ic_sup"] - t["cobertura"]]
    ax[0].barh(t["modelo"], t["cobertura"], xerr=err, color="#2b6cb0")
    ax[0].set(xlabel=f"Cobertura (z ≥ {Z_UMBRAL:g}, IC 95 %)", xlim=(0, 1))
    ax[1].barh(t["modelo"], t["tiempo_entrenamiento_s"], color="#718096")
    ax[1].set(xlabel="Tiempo de entrenamiento (s)")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Autoprueba del arnés
# MAGIC
# MAGIC Verifica las métricas con scores simulados (no son datos del proyecto; no escribe en MLflow ni en Unity Catalog):
# MAGIC scores al azar (uniformes o normales) ⇒ cobertura ≈ 0; un candidato que destaca 8 desviaciones en cada consulta ⇒ cobertura ≈ 1;
# MAGIC todo empatado ⇒ cobertura 0; reglas con un solo candidato con coincidencias ⇒ cubierta.

# COMMAND ----------

if AUTOPRUEBA:
    rng = np.random.default_rng(0)
    n_q, n_g = 300, 70
    pares = pd.DataFrame([(f"q{i}", f"f{j}") for i in range(n_q) for j in range(n_g)],
                         columns=["id_consulta", "id_ficha"])

    azar = pares.assign(score=rng.random(len(pares)))
    normal = pares.assign(score=rng.normal(size=len(pares)))
    destaca = normal.copy()
    destaca.loc[destaca["id_ficha"] == "f0", "score"] += 8
    empatado = pares.assign(score=1.0)
    reglas = pares.assign(score=(pares["id_ficha"] == "f0").astype(float) * 2)
    pocos = pares[pares["id_ficha"].isin(["f0", "f1", "f2"])].assign(score=1.0)

    m_a, _ = evaluar(azar)
    m_n, _ = evaluar(normal)
    m_d, _ = evaluar(destaca)
    m_e, _ = evaluar(empatado)
    m_r, _ = evaluar(reglas)
    m_p, _ = evaluar(pocos)
    assert m_a["cobertura"] == 0.0, m_a["cobertura"]
    assert m_n["cobertura"] < 0.05, m_n["cobertura"]
    assert m_d["cobertura"] > 0.95, m_d["cobertura"]
    assert m_e["cobertura"] == 0.0 and m_e["margen_mediano"] == 0.0
    assert m_r["cobertura"] == 1.0
    assert m_p["consultas_evaluables"] == 0 and m_p["consultas_pocos_candidatos"] == n_q
    print("Autoprueba OK — azar:", m_a["cobertura"], "| normal:", round(m_n["cobertura"], 3),
          "| destaca:", round(m_d["cobertura"], 3), "| empatado:", m_e["cobertura"], "| reglas:", m_r["cobertura"])
