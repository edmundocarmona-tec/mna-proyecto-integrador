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
# MAGIC | `evaluar()` | Recall@1/5/10/20, MRR, PR-AUC a nivel par, IC 95 % por bootstrap, desglose por nivel de ruido |
# MAGIC | `corrida()` | Run de MLflow con tags, parámetros, versiones Delta de las entradas y **tiempos de entrenamiento e inferencia** |
# MAGIC | `tabla_comparativa()` | Modelos ordenados por la métrica principal (Recall@10) con las métricas secundarias y los tiempos |
# MAGIC
# MAGIC **Dirección de la consulta (Decisión 20):** cada consulta es un registro de necropsia; la galería son fichas de búsqueda.
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
dbutils.widgets.dropdown("autoprueba", "false", ["true", "false"], "Correr autoprueba con datos sintéticos")

# COMMAND ----------

import time
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import average_precision_score

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
EXPERIMENTO = dbutils.widgets.get("experimento")
SEMILLA = int(dbutils.widgets.get("semilla"))
N_BOOTSTRAP = int(dbutils.widgets.get("n_bootstrap"))
AUTOPRUEBA = dbutils.widgets.get("autoprueba") == "true"

KS = [1, 5, 10, 20]
METRICA_PRINCIPAL = "recall_at_10"
METRICAS_SECUNDARIAS = ["mrr", "recall_at_1", "recall_at_5", "recall_at_20", "pr_auc_par"]

def _t(nombre):
    return f"{CATALOG}.{SCHEMA}.{nombre}"

# Entradas existentes
T_NECROPSIAS = _t("necropsias_enriquecidas")
T_FICHAS_TEXTO = _t("fichas_texto_embedding")
T_FICHAS_PERSONA = _t("fichas_persona_silver")
T_FICHAS_TATUAJES = _t("fichas_tatuajes_silver")
T_FICHAS_SENAS = _t("fichas_senas_silver")
T_FICHAS_VESTIMENTA = _t("fichas_vestimenta_silver")

# Salidas del Avance 4 (las crean los notebooks 11–14)
T_NEC_BRONZE = _t("necropsias_extraccion_bronze")
T_NEC_PERSONA = _t("necropsias_persona_silver")
T_NEC_TATUAJES = _t("necropsias_tatuajes_silver")
T_NEC_VESTIMENTA = _t("necropsias_vestimenta_silver")
T_NEC_PERTENENCIAS = _t("necropsias_pertenencias_silver")
T_NEC_TEXTO = _t("necropsias_texto_embedding")
T_PSEUDOFICHAS = _t("pseudofichas")              # galería sintética (estilo ficha)
T_VERDAD = _t("pseudofichas_verdad")             # (id_consulta, id_ficha, nivel_ruido, split)
T_CANDIDATOS = _t("candidatos_bloqueados")        # pares que sobreviven el blocking
T_RANKINGS = _t("resultados_rankings")           # rankings de cada run, con run_id

ENTRADAS_VERSIONADAS = [T_NECROPSIAS, T_FICHAS_TEXTO, T_FICHAS_VESTIMENTA, T_FICHAS_TATUAJES,
                        T_PSEUDOFICHAS, T_VERDAD, T_CANDIDATOS]

# COMMAND ----------

# MAGIC %md ## 2. MLflow: experimento y salvaguardas de privacidad

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
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
# MAGIC ## 4. Evaluación
# MAGIC
# MAGIC - **Empates:** se resuelven de forma **pesimista** (el correcto queda en el peor lugar del empate). Sin esto, un modelo
# MAGIC   de reglas con muchos scores iguales parecería mejor de lo que es.
# MAGIC - **Correcto fuera de los candidatos** (lo eliminó el blocking): cuenta como fallo. Así el costo del blocking entra en
# MAGIC   la métrica final y no se puede maquillar con un blocking estrecho.
# MAGIC - Solo se evalúan las consultas que tienen verdad conocida (pseudo-fichas); las necropsias reales sin match no entran.

# COMMAND ----------

def rango_del_correcto(rankings: pd.DataFrame, verdad: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila por consulta con el rango (1 = primero) de la ficha correcta, o NaN si no fue recuperada."""
    r = rankings[["id_consulta", "id_ficha", "score"]].copy()
    r["rango"] = r.groupby("id_consulta")["score"].rank(method="max", ascending=False)
    cols = ["id_consulta", "id_ficha"] + [c for c in ["nivel_ruido", "split"] if c in verdad.columns]
    out = verdad[cols].merge(r[["id_consulta", "id_ficha", "rango"]], on=["id_consulta", "id_ficha"], how="left")
    return out


def _metricas_de_rangos(rangos: np.ndarray) -> dict:
    m = {f"recall_at_{k}": float(np.mean(np.nan_to_num(rangos, nan=np.inf) <= k)) for k in KS}
    m["mrr"] = float(np.mean(np.nan_to_num(1.0 / rangos, nan=0.0)))
    return m


def _ic_bootstrap(rangos: np.ndarray, rng) -> dict:
    reps = {METRICA_PRINCIPAL: [], "mrr": []}
    n = len(rangos)
    for _ in range(N_BOOTSTRAP):
        m = _metricas_de_rangos(rangos[rng.integers(0, n, n)])
        reps[METRICA_PRINCIPAL].append(m[METRICA_PRINCIPAL])
        reps["mrr"].append(m["mrr"])
    ic = {}
    for nombre, valores in reps.items():
        ic[f"{nombre}_ic_inf"] = float(np.percentile(valores, 2.5))
        ic[f"{nombre}_ic_sup"] = float(np.percentile(valores, 97.5))
    return ic


def evaluar(rankings: pd.DataFrame, verdad: pd.DataFrame) -> dict:
    """Métricas agregadas de un ranking contra la verdad conocida, global y por nivel de ruido."""
    rng = np.random.default_rng(SEMILLA)
    rc = rango_del_correcto(rankings, verdad)
    rangos = rc["rango"].to_numpy(dtype=float)

    m = _metricas_de_rangos(rangos)
    m.update(_ic_bootstrap(rangos, rng))
    m["n_consultas"] = int(len(rc))
    m["recuperados_en_candidatos"] = float(np.mean(~np.isnan(rangos)))   # pair completeness efectiva
    m["candidatos_por_consulta"] = float(rankings.groupby("id_consulta").size().mean())

    # PR-AUC a nivel par (complemento): cada candidato es positivo si es la ficha correcta.
    pares = rankings.merge(verdad[["id_consulta", "id_ficha"]].assign(es_match=1),
                           on=["id_consulta", "id_ficha"], how="left").fillna({"es_match": 0})
    pares = pares[pares["id_consulta"].isin(verdad["id_consulta"])]
    m["pr_auc_par"] = float(average_precision_score(pares["es_match"], pares["score"])) if pares["es_match"].any() else 0.0

    if "nivel_ruido" in rc.columns:
        for nivel, g in rc.groupby("nivel_ruido"):
            for nombre, valor in _metricas_de_rangos(g["rango"].to_numpy(dtype=float)).items():
                m[f"{nombre}__ruido_{nivel}"] = valor
    return m, rc

# COMMAND ----------

# MAGIC %md ## 5. Gráficas (solo agregados)

# COMMAND ----------

def graficas_agregadas(rc: pd.DataFrame, titulo: str):
    rangos = rc["rango"].to_numpy(dtype=float)
    ks = np.arange(1, 51)
    curva = [np.mean(np.nan_to_num(rangos, nan=np.inf) <= k) for k in ks]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(ks, curva, color="#2b6cb0")
    for k in KS:
        ax[0].axvline(k, color="#cbd5e0", lw=0.8, zorder=0)
    ax[0].set(title="Recall@k", xlabel="k", ylabel="Recall", ylim=(0, 1.02))

    finitos = rangos[~np.isnan(rangos)]
    ax[1].hist(np.clip(finitos, 1, 51), bins=np.arange(1, 53) - 0.5, color="#2b6cb0")
    ax[1].set(title=f"Rango del correcto (no recuperados: {int(np.isnan(rangos).sum())})",
              xlabel="rango (51 = >50)", ylabel="consultas")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Corrida de MLflow
# MAGIC
# MAGIC Uso desde un notebook de modelo:
# MAGIC ```python
# MAGIC with corrida("3_regresion_logistica", familia="supervisada_lineal", fase="comparacion",
# MAGIC              params={"C": 1.0}) as c:
# MAGIC     with c.entrenamiento():
# MAGIC         modelo.fit(X_train, y_train)
# MAGIC     with c.inferencia():
# MAGIC         rankings = puntuar(modelo, candidatos_prueba)
# MAGIC     c.registrar(rankings, verdad_prueba)
# MAGIC ```
# MAGIC Los modelos sin entrenamiento (reglas SQL, BM25) registran como "entrenamiento" la construcción de su índice o
# MAGIC estadísticas; si no hay nada que construir, el tiempo es 0.

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

    def registrar(self, rankings: pd.DataFrame, verdad: pd.DataFrame, guardar_rankings=True):
        faltan = {"id_consulta", "id_ficha", "score"} - set(rankings.columns)
        assert not faltan, f"El ranking no cumple el contrato; faltan columnas: {faltan}"
        rankings = rankings[["id_consulta", "id_ficha", "score"]]

        metricas, rc = evaluar(rankings, verdad)
        n = max(metricas["n_consultas"], 1)
        metricas.update(self.tiempos)
        metricas["tiempo_inferencia_ms_por_consulta"] = 1000 * self.tiempos["tiempo_inferencia_s"] / n
        mlflow.log_metrics(metricas)
        mlflow.log_figure(graficas_agregadas(rc, self.nombre), "graficas/recall_y_rango.png")
        plt.close("all")

        if guardar_rankings:
            (spark.createDataFrame(rankings)
                  .selectExpr("CAST(id_consulta AS STRING) id_consulta", "CAST(id_ficha AS STRING) id_ficha",
                              "CAST(score AS DOUBLE) score")
                  .withColumn("run_id", F.lit(self.run.info.run_id))
                  .withColumn("modelo", F.lit(self.nombre))
                  .write.mode("append").saveAsTable(T_RANKINGS))

        print(f"{self.nombre}: Recall@10 = {metricas['recall_at_10']:.3f} "
              f"[{metricas['recall_at_10_ic_inf']:.3f}–{metricas['recall_at_10_ic_sup']:.3f}], "
              f"MRR = {metricas['mrr']:.3f}, entrenamiento = {self.tiempos['tiempo_entrenamiento_s']:.1f} s")
        return metricas


@contextlib.contextmanager
def corrida(nombre, familia, fase, params=None, nivel_ruido="todos", padre=None, tags=None):
    """Abre un run (anidado si hay un run activo o se da `padre`) con tags y parámetros comunes."""
    anidado = padre is not None or mlflow.active_run() is not None
    with mlflow.start_run(run_name=nombre, nested=anidado,
                          parent_run_id=padre) as run:
        mlflow.set_tags({"modelo": nombre, "familia": familia, "fase": fase, "nivel_ruido": nivel_ruido,
                         "direccion": "necropsia_a_fichas", **(tags or {})})
        mlflow.log_params({"semilla": SEMILLA, "metrica_principal": METRICA_PRINCIPAL, **(params or {})})
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
# MAGIC Tabla para la rúbrica: modelos **ordenados por la métrica principal**, con métricas secundarias y tiempos.

# COMMAND ----------

def tabla_comparativa(fase="comparacion", nivel_ruido="todos"):
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENTO],
        filter_string=f"tags.fase = '{fase}' AND tags.nivel_ruido = '{nivel_ruido}' AND tags.es_padre IS NULL",
    )
    if runs.empty:
        print("Sin corridas para", fase, nivel_ruido)
        return runs
    cols = (["tags.modelo", "tags.familia", f"metrics.{METRICA_PRINCIPAL}",
             f"metrics.{METRICA_PRINCIPAL}_ic_inf", f"metrics.{METRICA_PRINCIPAL}_ic_sup"]
            + [f"metrics.{m}" for m in METRICAS_SECUNDARIAS]
            + ["metrics.recall_at_10__ruido_alto", "metrics.tiempo_entrenamiento_s",
               "metrics.tiempo_inferencia_ms_por_consulta", "run_id"])
    cols = [c for c in cols if c in runs.columns]
    t = (runs[cols].sort_values(f"metrics.{METRICA_PRINCIPAL}", ascending=False)
                    .rename(columns=lambda c: c.split(".", 1)[-1]))
    # Si hay varias corridas del mismo modelo, se conserva la mejor.
    return t.drop_duplicates("modelo").reset_index(drop=True)


def grafica_comparativa(t: pd.DataFrame, titulo="Comparativa de modelos"):
    t = t.iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.5 * len(t) + 1.5), sharey=True)
    err = [t[METRICA_PRINCIPAL] - t[f"{METRICA_PRINCIPAL}_ic_inf"], t[f"{METRICA_PRINCIPAL}_ic_sup"] - t[METRICA_PRINCIPAL]]
    ax[0].barh(t["modelo"], t[METRICA_PRINCIPAL], xerr=err, color="#2b6cb0")
    ax[0].axvline(0.97, color="#c53030", ls="--", lw=1, label="meta 0.97")
    ax[0].set(xlabel="Recall@10 (IC 95 %)", xlim=(0, 1))
    ax[0].legend(loc="lower right")
    ax[1].barh(t["modelo"], t["tiempo_entrenamiento_s"], color="#718096")
    ax[1].set(xlabel="Tiempo de entrenamiento (s)")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Autoprueba con datos sintéticos
# MAGIC
# MAGIC Verifica el arnés sin tocar datos reales: un ranking perfecto debe dar Recall@k = 1 y MRR = 1, y uno al azar debe
# MAGIC acercarse a k / tamaño de galería. No escribe en MLflow ni en Unity Catalog.

# COMMAND ----------

from pyspark.sql import functions as F

if AUTOPRUEBA:
    rng = np.random.default_rng(0)
    n_q, n_g = 200, 100
    verdad_sint = pd.DataFrame({"id_consulta": [f"q{i}" for i in range(n_q)],
                                "id_ficha": [f"f{i % n_g}" for i in range(n_q)],
                                "nivel_ruido": rng.choice(["bajo", "medio", "alto"], n_q)})
    pares = pd.DataFrame([(q, f"f{j}") for q in verdad_sint["id_consulta"] for j in range(n_g)],
                         columns=["id_consulta", "id_ficha"])
    pares = pares.merge(verdad_sint[["id_consulta", "id_ficha"]].assign(ok=1), how="left").fillna({"ok": 0})

    perfecto = pares.assign(score=pares["ok"])
    azar = pares.assign(score=rng.random(len(pares)))
    empatado = pares.assign(score=1.0)                      # todo empatado: pesimista ⇒ rango = n_g

    m_p, _ = evaluar(perfecto, verdad_sint)
    m_a, _ = evaluar(azar, verdad_sint)
    m_e, _ = evaluar(empatado, verdad_sint)
    assert m_p["recall_at_1"] == 1.0 and m_p["mrr"] == 1.0
    assert abs(m_a["recall_at_10"] - 10 / n_g) < 0.06, m_a["recall_at_10"]
    assert m_e["recall_at_20"] == 0.0, "los empates deben resolverse de forma pesimista"
    print("Autoprueba OK — perfecto:", m_p["recall_at_10"], "| azar:", round(m_a["recall_at_10"], 3),
          "| empatado:", m_e["recall_at_20"])
