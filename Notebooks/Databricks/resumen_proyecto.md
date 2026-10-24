# Resumen del Proyecto Forensic Matching

## Contexto general

**Proyecto:** Cotejo ante-mortem / post-mortem de personas desaparecidas en Querétaro, México. Parte del Proyecto Integrador MNA (Tec de Monterrey, Equipo 23).

**Objetivo:** Comparar fichas de búsqueda de personas desaparecidas (datos ante-mortem, extraídos con LLM de PDFs/imágenes) contra registros de necropsias (datos post-mortem, en Excel) para proponer identificaciones por similitud semántica usando Vector Search.

**Infraestructura Databricks:**
- **Catalog:** `proyecto_mna`
- **Schema:** `fichas_busqueda`
- **Volume:** `fichas_raw` (64 archivos: 45 PDFs + 19 imágenes)
- **Vector Search Endpoint:** `forensic` (ONLINE)
- **Embedding model:** `databricks-qwen3-embedding-0-6b` (multilingüe, español)
- **LLM endpoint:** `proyecto_mna.fichas_busqueda.gema` (AI Gateway, Claude Sonnet 4.5)
- **Compute:** Serverless interactivo (Spark Connect)

---

## Notebook 01: Extracción de fichas con LLM (`01_extraccion_fichas_llm`)

**Propósito:** Convertir cada ficha de búsqueda (PDF o imagen con texto + foto en vida) en un registro estructurado con esquema fijo usando un LLM multimodal.

**Pipeline:**
1. Lee archivos del volumen (`/Volumes/proyecto_mna/fichas_busqueda/fichas_raw`)
2. Renderiza cada página de PDF a PNG (150 DPI con PyMuPDF)
3. Envía TODAS las páginas de un PDF en una sola llamada al LLM (fix v2: antes procesaba página por página y alucinaba fichas separadas)
4. El LLM devuelve JSON con esquema Pydantic (enums controlados: sexo, complexion, color_piel, etc.)
5. `SIN_DATO` distingue "no mencionado" de un valor real
6. Guarda en tablas Delta: bronze (JSON crudo), silver (persona, tatuajes, señas, vestimenta), texto_embedding (texto canónico por facet)

**Tablas que produce:**
- `fichas_extraccion_bronze` — JSON crudo de cada extracción
- `fichas_persona_silver` — 60 filas, datos estructurados (sexo, edad, estatura, peso, media filiación)
- `fichas_tatuajes_silver` — 100 filas, tatuajes detallados por ficha
- `fichas_senas_silver` — 25 filas, señas particulares
- `fichas_vestimenta_silver` — 82 filas, vestimenta
- `fichas_texto_embedding` — 60 filas, texto canónico por facet: `texto_rasgos`, `texto_tatuajes`, `texto_senas`, `texto_vestimenta`, `texto_completo`

**Estado:** Funcionando (v2). 64 archivos → 60 extracciones OK + 4 errores de rate limit (imágenes de WhatsApp). El LLM no transcribe nombres; usa `id_ficha` (SHA-256 del folio) como seudónimo.

**Parámetros configurables (widgets):** catalog, schema, volume, endpoint, prompt_version (v2), dpi, max_workers, limit, strict_schema.

---

## Notebook 02: EDA de fichas (`02_eda_fichas_busqueda`)

**Propósito:** Análisis exploratorio de `fichas_texto_embedding` joined con `fichas_persona_silver`.

**Secciones (21 celdas):**
1. **Faltantes** — % de NULL/vacío/SIN_DATO por columna. Cruzado con estado_hechos y rango_edad.
2. **Estadísticas** — Numéricas (edad 15-66, estatura 149-180, peso 45-95) y categóricas (valor modal + %).
3. **Atípicos** — IQR por rango_edad para estatura y peso. Rangos válidos: estatura 40-220cm, peso 2-250kg. No se eliminan, solo se marcan.
4. **Cardinalidad** — Valores distintos por columna categórica. Identifica variantes de escritura (ej: "MEXICO" = "ESTADO DE MEXICO").
5. **Sesgo** — skewness: edad 0.59 (moderado), estatura -0.15, peso 0.37.
6. **Tendencias** — temporales.
7. **Correlación/Entropía** — Shannon entropy excluyendo SIN_DATO.
8. **Bivariado** — relaciones entre variables.
9. **Imágenes** — N/A.
10. **Desequilibrio** — balance de clases.

**Hallazgos clave:**
- Faltantes: cabello_tipo 36.7% SIN_DATO, nariz 43.3% SIN_DATO, estatura 20% NULL.
- 76.7% con tatuajes, 28.3% con señas, 51.7% con vestimenta.
- Calidad foto: REGULAR 43, BUENA 17.

**Estado:** Ejecutado. Útil para entender calidad de datos antes de mejorar el matching.

---

## Notebook 03: Feature Engineering (`03 Feature Engineering`)

**Propósito:** Preparación de datos según CRISP-ML(Q). La técnica es **retrieval por similitud** (Vector Search k-NN + blocking), NO clasificación supervisada.

**Rol triple de la FE:**
1. Seleccionar qué features van al texto canónico que se embebe
2. Definir criterios de blocking como metadata de Vector Search (filter_sexo, filter_grupo_edad, filter_rango_estatura, filter_region_geografica)
3. Análisis exploratorio (PCA/FA revelan estructura latente)

**Secciones (24 celdas):**
1. **Generación de features nuevas:** IMC (peso/estatura²), indicadores binarios (has_tatuajes, has_senas, has_vestimenta), región geográfica (mapeo estado→región), días desde hechos, densidad de tatuajes, n_regiones_tat, tat_categoria_top, n_tipos_vestimenta.
2. **Discretización (binning):** grupo_edad (infante_adolescente, joven_adulto, adulto_joven, adulto, adulto_mayor), categoría_imc, rango_estatura (bajo/medio/alto), antigüedad_caso.
3. **Codificación:** Ordinal (labios DELGADOS<MEDIANOS<GRUESOS) y One-Hot/Nominal.
4. **Selección de features:** VarianceThreshold, SelectKBest, RFE, análisis de importancia.
5. **Texto canónico:** Construye `texto_canonico_fe` en lenguaje natural con features que pasaron selección.

**Estado:** Ejecutado. Genera la tabla gold que alimenta los índices de Vector Search.

---

## Notebook 04: Consultas Vector Search (`04 Vector Search Queries`)

**Propósito:** Demostrar cómo consultar el índice `texto_completo_sin_features` para recuperar fichas por similitud semántica.

**Índice usado:** `proyecto_mna.fichas_busqueda.texto_completo_sin_features`
- Delta Sync, TRIGGERED, endpoint `forensic`
- PK: `id_ficha`, embedding en `texto_completo` (managed, databricks-qwen3-embedding-0-6b)
- Columnas synced: id_ficha, sexo, edad_desaparicion, estado_hechos, municipio_hechos, texto_rasgos, texto_tatuajes, texto_senas, texto_vestimenta, texto_completo

**6 celdas:**
1. **Setup** — WorkspaceClient + configuración del índice
2. **k-NN básica** — query de texto sin filtros, top-5
3. **k-NN con blocking** — filtros por sexo
4. **`buscar_candidatos(query_text, k=5, filters=None)`** — función reutilizable que devuelve DataFrame de pandas
5. **Comparación necropsias vs fichas** — carga `DESCRIPCIÓN TATUAJES` de `registro_necropsias_2026`, mapea sexo, hace blocking, join con `fichas_persona_silver` para rutas de imagen, muestra `texto_tatuajes` + `edad_desaparicion` + `estado_hechos`
6. **Galería visual** — thumbnails de imágenes del volumen con PIL + displayHTML

**Estado:** Funcionando. Es la base para las consultas de matching.

---

## Notebook 05a: Cruce de necropsias enriquecidas (`05_cruce_necropsias_enriquecidas`)

**Propósito:** Cruzar dos tablas de necropsias para enriquecer los datos post-mortem.

**Tablas que cruza:**
- `registro_necropsias_2020_2026` (NUEVA): 1,246 filas, años 2006-2026. Superconjunto. Columnas toscas, sin columna de tatuajes explícita.
- `registro_necropsias_2026` (VIEJA): 58 filas. Subconjunto completo de la nueva. Detalle forense: tatuajes, cicatrices, edad, municipio, genética, causa de muerte.

**Estrategia:** LEFT JOIN nueva→vieja ON `No`=`#`. Conserva las 1,246 filas y enriquece las 58 compartidas.

**7 celdas:**
1. Markdown título
2. Leer y renombrar columnas sucias (saltos de línea \n, espacios finales)
3. Validar traslape (58 coinciden, 0 solo en vieja, 1,188 solo en nueva)
4. LEFT JOIN + `texto_canonico_nec` (ropa|pertenencias|sintesis) + `tiene_tatuaje_bin` (fallback ILIKE %TATUAJE%)
5. Normalización de sexo: `sexo_norm` (preferir sexo_viejo HOMBRE/MUJER, else normalizar sexo_nec). Audita 6 discrepancias.
6. Guarda en `proyecto_mna.fichas_busqueda.necropsias_enriquecidas` (1,246 filas, 28 columnas)
7. Notas y próximos pasos

**Resultado:** `necropsias_enriquecidas` — 1,246 filas. tiene_tatuaje_bin: SI=12/NO=1234. sexo_norm: HOMBRE 1085, MUJER 104, INDETERMINADO 57.

**Estado:** Ejecutado y funcionando.

---

## Notebook 05b: Matching de tatuajes (`05_matching`)

**Propósito:** Cotejo de tatuajes: descripciones de necropsias vs fichas de búsqueda vía Vector Search.

**4 celdas:**
1. **Setup** — carga 9 necropsias con descripciones reales de tatuajes de `registro_necropsias_2026` (IDs: 13, 32, 105, 215, 249, 301, 402, 584, 596)
2. **Consulta Vector Search** — para cada necropsia, usa `DESCRIPCIÓN TATUAJES` como query_text con blocking por sexo. Función `buscar_candidatos_tatuajes()`.
3. **Tabla comparativa** — 90 comparaciones (9 necropsias × top-10), muestra tatuajes lado a lado.

**Resultados actuales:** Scores 0.52-0.59. Las mismas fichas aparecen como top candidates para diferentes necropsias, lo que sugiere que el modelo capta patrones de texto general en lugar de similitud específica de tatuajes.

**Problema identificado:** El índice `texto_completo_sin_features` embebe `texto_completo` (texto completo), no `texto_tatuajes` específicamente. Los tatuajes son solo una fracción del texto total, diluyendo la señal.

**Estado:** Funcionando pero con resultados subóptimos. Es la referencia del baseline actual.

---

## Notebook 06: Multi-facet Vector Search (`06_multi_facet_vector_search`)

**Propósito:** Crear 3 índices separados de Vector Search, cada uno embebiendo una faceta forense diferente, para resolver el problema de dilución de señal del notebook 05b.

**Tabla gold:** `proyecto_mna.fichas_busqueda.fichas_vs_gold_v2` (60 filas, CDF=true)
- Columnas: id_ficha, filter_sexo, filter_grupo_edad, filter_estado, edad_desaparicion, estatura_cm, peso_kg, texto_tatuajes, texto_vestimenta, texto_senas
- Cobertura: 46 tatuajes (77%), 31 vestimenta (52%), 17 señas (28%)

**3 índices creados:**

| Índice | Embedding en | Peso forense | Cobertura |
|--------|-------------|-------------|----------|
| `vs_tatuajes_idx` | texto_tatuajes | 0.65 | 77% |
| `vs_vestimenta_idx` | texto_vestimenta | 0.25 | 52% |
| `vs_senas_idx` | texto_senas | 0.10 | 28% |

- Todos con blocking: filter_sexo (hard), filter_grupo_edad (soft), filter_estado (soft)
- Todos en endpoint `forensic`, modelo `databricks-qwen3-embedding-0-6b`, TRIGGERED pipeline

**Flujo de consulta propuesto:** necropsia → consultar cada índice de faceta relevante (tatuajes si descr_tatuajes, vestimenta si ropa, señas si cicatriz) → combinar scores ponderados (0.65/0.25/0.10, re-normalizados si faltan facetas) → aplicar blocking → re-rank con features estructuradas (edad, estatura, estado).

**Estado:** Los 3 índices fueron creados exitosamente. El test de tatuajes falló (`w` no definido — error de sesión). El test de vestimenta no se ejecutó. **Falta: implementar el query multi-facet con weighted re-ranking y validarlo.**

---

## Estado actual y qué falta (para fine-tuning)

**Lo que funciona:**
- Extracción LLM de fichas (notebook 01) — 60 fichas extraídas
- EDA (notebook 02) — calidad de datos entendida
- Feature engineering (notebook 03) — texto canónico + blocking filters
- Consultas básicas de Vector Search (notebook 04) — `buscar_candidatos()` funcional
- Enriquecimiento de necropsias (notebook 05a) — 1,246 filas enriquecidas
- Índices multi-facet creados (notebook 06) — 3 índices ONLINE

**Lo que falta / necesita mejora:**
1. **Implementar query multi-facet** (notebook 06): el weighted re-ranking con los 3 índices no está implementado ni validado. Solo se crearon los índices.
2. **Mejorar scores de matching**: actualmente 0.52-0.59 con un solo índice. El multi-facet debería mejorar esto al dar más peso a tatuajes (la señal más discriminativa).
3. **Escalar el matching**: el notebook 05b solo prueba con 9 necropsias. Hay 1,246 en la tabla enriquecida (892 NO IDENTIFICADO). Falta correr el matching a escala.
4. **Fine-tuning del embedding**: el modelo `databricks-qwen3-embedding-0-6b` es genérico. Considerar fine-tuning con datos forenses para mejorar la captura de similitud en descripciones de tatuajes.
5. **Normalización de edad en necropsias**: la edad viene como texto ("40-50 AÑOS", "30-35 AÑOS") y necesita convertirse a rango numérico para blocking.
6. **Evaluar resultados**: definir métricas de evaluación (Recall@k, MRR, nDCG) y ground truth para medir mejoras.