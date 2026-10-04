# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Proyecto Integrador (MNA — Maestría en Inteligencia Artificial Aplicada) building a matching system between
**fichas de búsqueda** (ante-mortem data on missing persons) and the **libro de registro de necropsias**
(post-mortem forensic data on unidentified bodies) for the state of Querétaro. The goal is a retrieval/ranking
pipeline that proposes likely candidate matches to speed up human identification — not a classifier or regression
model with a single ground-truth label.

This is a data/notebook-driven research repo, not a software package: there is no build, lint, or test tooling.
Work happens in Databricks notebooks (`Notebooks/`) against raw data (`Datos/`) and is written up in
`Documentación/`.

## Repository structure

- `README.md` — project description, objectives, team.
- `Datos/` — raw inputs: the necropsias/reconocimientos registry (`.xlsx`) and sample ficha images (`.jpeg`).
  Contains sensitive personal data — see Data sensitivity below.
- `Notebooks/` — Databricks notebooks (committed as `.py` files using the `# Databricks notebook source` /
  `# COMMAND ----------` cell-marker format, or as `.ipynb`). Numbered in pipeline order.
- `Documentación/` — write-ups and deliverables.

## Pipeline architecture

The pipeline runs inside Databricks (Spark + Unity Catalog), not locally. Key stages, each a numbered notebook:

1. **`01_extraccion_fichas_llm.py`** — Extracts structured data from ficha images/PDFs using a multimodal LLM
   (via an OpenAI-compatible client pointed at Databricks' AI Gateway):
   - Renders each PDF page / image to PNG (PyMuPDF), calls the LLM with a **fixed Pydantic schema**
     (`FichaExtraida`), and validates/retries on failure.
   - All categorical fields use **controlled vocabularies** (`Literal` enums) with a universal `SIN_DATO`
     sentinel distinguishing "not mentioned" from a real value — this keeps both sides of the eventual
     ante-mortem/post-mortem comparison on the same vocabulary and keeps embedding input low-cardinality.
   - A normalization layer (`normalizar_salida` / `SINONIMOS` dict) maps free-text synonyms the LLM might still
     emit (e.g. "aguileña", "chamorro", "porlicue") onto the controlled vocabulary before validation.
   - Writes a `bronze` table (raw LLM JSON per page, keyed by image `sha256` + model + prompt fingerprint, so
     reprocessing is idempotent) then flattens into `silver` tables: `fichas_persona_silver` (1 row/ficha),
     and child tables `fichas_tatuajes_silver`, `fichas_senas_silver`, `fichas_vestimenta_silver` (1:N).
   - `id_ficha` is a SHA-256 hash of the folio (or of the image hash if no folio) — a stable pseudonym used to
     join tables **without ever storing the person's name**.
   - Builds deterministic canonical text per facet (`texto_rasgos`, `texto_tatuajes`, `texto_senas`,
     `texto_vestimenta`, `texto_completo`) from the structured fields — this is the input for embeddings /
     Databricks Vector Search, intentionally *not* raw LLM free text.
   - Genetic profile and fingerprint data (signals 4–5) are explicitly **not** extracted by the LLM — they are
     primary identifiers that come from the lab/AFIS and are loaded as separate structured tables
     (`perfil_genetico`, `huellas_referencia`); the LLM only records whether a ficha *mentions* they exist.
2. **`02_eda_fichas_busqueda.ipynb`** — EDA over the `silver`/text tables: joins `fichas_persona_silver` with
   `fichas_texto_embedding`, profiles missingness, categorical cardinality/variants (state/municipality spelling
   variants via `difflib.get_close_matches`), outliers in estatura/peso (IQR-based, not removed, just flagged),
   and temporal distribution of `fecha_hechos`. Explicitly treats this as a **retrieval** problem: numeric fields
   (edad, estatura, peso) are meant to become tolerance-window filters (e.g. ±5 years, ±10 cm) in the matching
   step, not regression targets — no log/Box-Cox transforms are applied despite skew.

Planned next steps referenced in the notebooks: manual validation of LLM extraction accuracy, applying the same
schema/pipeline to the post-mortem necropsias book, then building embeddings + a Databricks Vector Search index
with blocking filters (sexo, rango de edad, fecha/estado) to actually cross-match the two sides.

## Working with the notebooks

- Notebooks are meant to run **inside Databricks**, not as standalone local Python — they use `dbutils`,
  `spark`, `display()`, and Unity Catalog three-level names (`catalog.schema.table`). There is no local
  runner/test harness for them in this repo.
- Parameters (catalog/schema/volume names, model endpoint, prompt version, DPI, concurrency, strict-schema mode)
  are Databricks widgets at the top of notebook 01 — change behavior by editing widget defaults, not by hardcoding
  further down.
- The LLM call path is schema-first: if you need a new extracted field, add it to the relevant Pydantic model
  (`MediaFiliacion`, `Tatuaje`, `SenaParticular`, `Prenda`, or `FichaExtraida`), not by post-hoc parsing of free
  text. Any new categorical field should be a `Literal[...]` with a `SIN_DATO` fallback, following the existing
  pattern, and new synonyms should be added to `SINONIMOS` rather than special-cased in the prompt.
- The prompt fingerprint (`PROMPT_FINGERPRINT`, hash of system prompt + JSON schema) drives idempotency — bronze
  rows are only reprocessed when the prompt or schema actually changes, so editing `SYSTEM_PROMPT` or any Pydantic
  model intentionally invalidates the cache for affected pages.

## Data sensitivity

This project handles sensitive personal data about missing and deceased persons under agreement with state
authorities (Comisión Local de Búsqueda / Fiscalía de Querétaro). Rules already encoded in the pipeline that must
be preserved in any changes:

- Never transcribe or extract a person's **name** or contact info into any field — the join key across tables is
  the pseudonymized `id_ficha`, never a name.
- Genetic profile and fingerprint data are primary identifiers and are **never** passed through the LLM; they are
  loaded as separate structured tables and only referenced (not derived) elsewhere.
- `bronze`/`silver` tables are documented as requiring restricted-permission schemas (Unity Catalog) — don't
  suggest storing this data in less-restricted locations.
- The LLM must not invent/guess missing values; absent data is `SIN_DATO`/`null`, never an inferred placeholder.
