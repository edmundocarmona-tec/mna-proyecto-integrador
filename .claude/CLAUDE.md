# Proyecto Integrador MNA — Cotejo de fichas de búsqueda vs. registros forenses (Querétaro)

Equipo 23, Grupo 10. El contexto completo del proyecto está en el skill
`proyecto-integrador-identificacion-forense` (`.claude/skills/`). Cárgalo antes
de trabajar en cualquier notebook, documento o experimento de este repo.

## Reglas que no se reabren sin que el autor lo pida
- Problema de **recuperación por similitud / record linkage**, no clasificación supervisada.
- Pipeline: extracción con LLM multimodal y esquema fijo → embeddings → Databricks Vector Search con blocking (sexo, edad, fechas, municipio) → ranking + justificación. El modelo sugiere, el perito confirma.
- Fase 1 = solo descripciones textuales. Foto-a-foto y superposición craneofacial son trabajo futuro.
- Métricas principales: Recall@k (1/5/10/20) y MRR. PR-AUC a nivel par solo como complemento.

## Privacidad (obligatorio)
- Las fichas y el libro de necropsias contienen identidades reales.
- Nunca imprimir, registrar ni commitear filas reales: nombres, folios, carpetas de investigación, fotos o textos de fichas individuales.
- En MLflow: solo parámetros, métricas agregadas y gráficas de agregados. No `log_input` con contenido ni datasets crudos como artefactos.
- Antes de cada commit, escanear salidas de notebooks en busca de nombres y carpetas de investigación.

## Estado actual
- **Avance 2 (EDA fichas):** notebook `01_extraccion_fichas_llm.py` (esquema Pydantic con vocabularios controlados, normalización de sinónimos, endpoint `proyecto_mna.fichas_busqueda.gema` vía Responses API del AI Gateway). Tatuajes y señas son los rasgos más discriminantes.
- **Avance 3 (baseline):** pseudo-fichas con ruido controlado, partición agrupada por ficha. B0 azar Recall@10 = 0.028; B1 reglas + TF-IDF = 0.850; **B2 regresión logística sobre vector de comparación Fellegi-Sunter = 0.869** (IC 95 % 0.804–0.925; con ruido alto 0.710). Blocking: mismo sexo y |Δedad| ≤ 15 cuando ambos datos existen.
- **Meta del modelo avanzado:** Recall@10 ≥ 0.97 y ≥ 0.80 con ruido alto.

## Avance 4 (en curso)
Tarea: ≥ 6 modelos individuales (no ensambles), comparar, elegir los 2 mejores, ajustarlos y elegir el final. Se comparan **tipos de búsqueda** sobre el mismo conjunto de candidatos bloqueado, registrando todo en **MLflow (Databricks)**:

1. Léxica: BM25 por facetas
2. Probabilística no supervisada: Fellegi-Sunter con EM (Splink)
3. Supervisada lineal: regresión logística (B2)
4. Learning-to-rank: LambdaMART (LightGBM `lambdarank`)
5. Semántica densa: embeddings multilingües + k-NN en Vector Search
6. Semántica de interacción: cross-encoder re-ranker

Evaluación: (A) pseudo-fichas en estilo ficha, (B) fichas reales con expediente conocido, (C) fichas reales sin match para revisión cualitativa. Ya hay fichas reales procesadas en Databricks.

## Declaración de uso de IA
`docs/declaracion_uso_ia.md` es un documento vivo (formato Decisión / Justificación / Alternativa descartada y motivo / Participación de la IA). Agrega una decisión nueva cada vez que se tome una decisión metodológica.

## Repo
GitHub: `edmundocarmona-tec/mna-proyecto-integrador` (notebooks en `Notebooks/`).

## No mezclar
El curso TC5038 (caso Marriott) es otro proyecto. No usar su contexto aquí.
