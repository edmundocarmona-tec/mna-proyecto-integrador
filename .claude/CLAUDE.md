# Proyecto Integrador MNA — Cotejo de fichas de búsqueda vs. registros forenses (Querétaro)

Equipo 23, Grupo 10. El contexto completo del proyecto está en el skill
`proyecto-integrador-identificacion-forense` (`.claude/skills/`). Cárgalo antes
de trabajar en cualquier notebook, documento o experimento de este repo.

## Reglas que no se reabren sin que el autor lo pida
- Problema de **recuperación por similitud / record linkage**, no clasificación supervisada.
- Pipeline: extracción con LLM multimodal y esquema fijo → embeddings → Databricks Vector Search con blocking (sexo, edad, fechas, municipio) → ranking + justificación. El modelo sugiere, el perito confirma.
- Fase 1 = solo descripciones textuales. Foto-a-foto y superposición craneofacial son trabajo futuro.
- **Cotejo NO supervisado (restricción del proyecto).** No existen pares ficha ↔ necropsia confirmados: no hay verdad conocida. Por eso:
  - No se entrenan modelos supervisados (regresión logística, LambdaMART ni ningún clasificador).
  - No se generan datos sintéticos ni pseudo-fichas.
  - No se vuelven a extraer fichas ni necropsias con LLM: se trabaja con las tablas reales que ya existen.
  - Recall@k y MRR no se pueden calcular. Solo volverían a ser las métricas principales si el equipo consigue pares confirmados.

## Privacidad (obligatorio)
- Las fichas y el libro de necropsias contienen identidades reales.
- Nunca imprimir, registrar ni commitear filas reales: nombres, folios, carpetas de investigación, fotos o textos de fichas individuales.
- En MLflow: solo parámetros, métricas agregadas y gráficas de agregados. No `log_input` con contenido ni datasets crudos como artefactos.
- Antes de cada commit, escanear salidas de notebooks en busca de nombres y carpetas de investigación.

## Estado actual
- **Avance 2 (EDA fichas):** notebook `01_extraccion_fichas_llm.py` (esquema Pydantic con vocabularios controlados, normalización de sinónimos, endpoint `proyecto_mna.fichas_busqueda.gema` vía Responses API del AI Gateway). Tatuajes y señas son los rasgos más discriminantes.
- **Avance 3 (baseline):** pseudo-fichas con ruido controlado, partición agrupada por ficha. B0 azar Recall@10 = 0.028; B1 reglas + TF-IDF = 0.850; **B2 regresión logística sobre vector de comparación Fellegi-Sunter = 0.869** (IC 95 % 0.804–0.925; con ruido alto 0.710). Blocking: mismo sexo y |Δedad| ≤ 15 cuando ambos datos existen.
- **Meta del Avance 3** (Recall@10 ≥ 0.97 con pseudo-fichas): no aplica al Avance 4, que no tiene verdad conocida.

## Avance 4 (en curso)
Tarea: ≥ 6 modelos individuales (no ensambles), comparar, elegir los 2 mejores, ajustarlos y elegir el final. Todo se registra en **MLflow (Databricks)**: experimento `/Shared/proyecto_mna/avance4_cotejo`, con el arnés `Notebooks/Databricks/00_config_evaluacion.py`. Todos los modelos son **no supervisados** y corren sobre el mismo conjunto de candidatos bloqueado (mismo sexo):

0. Baseline determinístico: reglas tipo SQL (coincidencia exacta de prenda + color, marca, tatuaje)
1. Cadenas aproximadas: Jaro-Winkler / Levenshtein
2. Léxica: TF-IDF + coseno
3. Léxica ponderada: BM25 por facetas
4. Probabilística: Fellegi-Sunter con EM (Splink)
5. Semántica densa: embeddings multilingües + k-NN en Vector Search
6. Semántica de interacción: cross-encoder preentrenado (zero-shot, sin ajustarlo con nuestros datos)

Datos (ya existen, no se regeneran):
- Consultas: `necropsias_enriquecidas` (ropa, pertenencias, descr_tatuajes). Dirección necropsia → fichas.
- Galería: `fichas_vector_search_gold` (75 fichas, `texto_canonico_fe` y filtros).

"Ajuste fino" = hiperparámetros de cada método: pesos por faceta, k1/b de BM25, umbrales de Splink, modelo de embeddings.

Evaluación (sin etiquetas, Decisión 25): **métrica principal = cobertura** = % de necropsias evaluables (≥ 15 candidatos) cuyo mejor candidato destaca con z robusto ≥ 5 frente al resto de sus candidatos (mediana/MAD dentro de la consulta; umbral calibrado para que un método aleatorio dé ≤ 2.5 %). Secundarias: margen mediano top-1 vs top-2, cobertura con z ≥ 2, concordancia top-10 entre métodos, tiempos de entrenamiento e inferencia. La cobertura mide señal, no acierto: la confirmación es pericial (no disponible por ahora).

## Declaración de uso de IA
`docs/declaracion_uso_ia.md` es un documento vivo (formato Decisión / Justificación / Alternativa descartada y motivo / Participación de la IA). Agrega una decisión nueva cada vez que se tome una decisión metodológica.

## Repo
GitHub: `edmundocarmona-tec/mna-proyecto-integrador` (notebooks en `Notebooks/`).

## No mezclar
El curso TC5038 (caso Marriott) es otro proyecto. No usar su contexto aquí.
