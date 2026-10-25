# Plan — Avance 4: cotejo por facetas necropsia → fichas

> **AVISO (Decisión 24): el Avance 4 es un cotejo NO supervisado.** No hay pares ficha ↔ necropsia confirmados.
> No se entrenan modelos supervisados, no se generan pseudo-fichas ni datos sintéticos, y no se vuelve a extraer
> nada con LLM. Se compara directamente `necropsias_enriquecidas` (consultas) contra `fichas_vector_search_gold`
> (galería, 75 fichas). Las secciones de este documento sobre extracción de necropsias (P1), pseudo-fichas (P2),
> modelos supervisados (regresión logística, LambdaMART) y Recall@k quedan **descartadas**; ver `CLAUDE.md`.


Proyecto Integrador MNA — Equipo 23. Documento de trabajo (24 de octubre de 2026).
Todo se ejecuta en Databricks sobre `proyecto_mna.fichas_busqueda`. Ninguna fila real sale del workspace.

## 1. Pregunta que responde el sistema

> Para cada persona registrada en el libro de necropsias, ¿qué fichas de búsqueda son los candidatos más probables,
> y por qué (qué tatuajes, prendas o señas coinciden)?

- **Consulta:** un registro de necropsia. **Galería:** las fichas de búsqueda.
- **Salida:** top-k fichas con score y justificación por faceta. El perito confirma.
- **Métricas:** Recall@1/5/10/20 y MRR por consulta; *pair completeness* del blocking; PR-AUC a nivel par solo como complemento.

## 2. Insumos

| Tabla | Filas | Qué aporta | Problema |
|---|---|---|---|
| `fichas_texto_embedding` (+ `fichas_*_silver`) | 60 | Facetas estructuradas con vocabulario controlado (tatuajes, señas, vestimenta, rasgos) | Galería pequeña; fichas públicas, no necesariamente de Querétaro |
| `registro_necropsias_2020_2026` | 1,246 | Sexo, status, ropa, pertenencias, síntesis, condiciones de ingreso | Texto libre, sin tatuajes/edad/municipio como columnas propias (por confirmar en el perfilado) |
| `registro_necropsias_2026` | 58 | Tatuajes, cicatrices, vestimenta, edad, municipio | Solo 58 casos; subconjunto de la anterior (`No = #`) |
| **`necropsias_enriquecidas`** (notebook 05a) | 1,246 | **Fuente consolidada post-mortem**: LEFT JOIN de las dos anteriores, con `sexo_norm`, `texto_canonico_nec` y `tiene_tatuaje_bin` | Columnas de detalle forense solo en 58 filas; `edad` sigue en texto |

P0 y P1 parten de `necropsias_enriquecidas`; las dos tablas originales solo se consultan si falta alguna columna.

**Asimetría central:** las fichas están estructuradas y las necropsias son texto libre. Comparar texto canónico contra
texto crudo explica los scores planos (0.52–0.59) del notebook `05_matching`. La solución es llevar las necropsias
al **mismo esquema** que las fichas antes de comparar.

### 2.1 Resultados de P0 (24 oct 2026, solo agregados, warehouse serverless)

**Necropsias (`necropsias_enriquecidas`, 1,246):**

| | NO IDENTIFICADO | IDENTIFICADO NO RECLAMADO |
|---|---|---|
| Registros | 892 | 354 |
| Ropa descrita (> 15 caracteres) | 596 (67 %) | 251 (71 %) |
| Pertenencias descritas | 248 | 103 |
| Tatuajes (campo o mención en texto) | 8 | 4 |
| Señas en texto (cicatriz, lunar, prótesis…) | 3 | 2 |
| Edad / municipio (solo tabla 2026) | 47 | 11 |
| Cuerpo degradado (putrefacción, esqueletización, calcinación) | 110 | 25 |

- Sexo: 1,085 hombres, 104 mujeres, 57 indeterminados. Tipo de ingreso: 1,077 cadáveres, 93 restos óseos, 61 fetos.
- No hay registros "IDENTIFICADO" reclamados en la tabla: los identificados son solo los no reclamados.
- `fecha_ingreso` está en prácticamente todos los registros (2006–2026).
- La ropa es texto semiestructurado y rico: tipo de prenda, color, marca, talla, estampado/leyenda, estado de conservación (~210 caracteres en promedio).

**Fichas (`fichas_texto_embedding`, 60):** tatuajes 46, vestimenta 31, señas 17; 59 con fecha de hechos (2005–2026);
43 en Querétaro; 45 hombres. Tipos de prenda más comunes: pantalón, calzado, playera, camisa.

**Consecuencias para el diseño:**
1. **La vestimenta es la faceta principal**: está en 68 % de las necropsias y en 52 % de las fichas. Los tatuajes, que son la faceta más discriminante de las fichas, casi no se registran del lado post-mortem (12 de 1,246). El emparejamiento de tatuajes queda como señal secundaria y como **recomendación de captura al SEMEFO**.
2. **El blocking por edad solo aplica a 58 registros.** Para el resto, el blocking es sexo + ventana temporal (`fecha_hechos ≤ fecha_ingreso`) + tipo de ingreso (los fetos se excluyen de la consulta).
3. **Las pertenencias** (351 registros) entran como faceta propia: documentos, celulares, joyería.
4. **Pseudo-fichas:** se generan desde los 251 identificados no reclamados que tienen ropa descrita.
5. **Evaluación (B):** sin nombres ni una tabla de identificados reclamados, no hay pares reales verificables dentro de estas tablas; depende de que el equipo consiga una lista de cotejos confirmados.

## 3. Pipeline

```
P0 perfilado ─► P1 extracción necropsias (LLM, mismo esquema) ─► P2 pseudo-fichas + partición
                                                                    │
                    P3 candidatos (blocking) ◄──────────────────────┘
                          │
                    P4 vector de comparación por faceta
                          │
       P5 seis modelos (MLflow) ─► P6 ajuste de los 2 mejores ─► P7 modelo final + salida para el perito
```

### P0 — Perfilado de las tablas de necropsias (solo agregados)
- Esquema completo de `registro_necropsias_2020_2026`: ¿hay columnas de edad, municipio o fechas no usadas en 05a?
- % de llenado por campo de texto (ropa, pertenencias, síntesis, tatuajes, cicatrices) y por `status`.
- Cuántos registros **NO IDENTIFICADO** tienen al menos una faceta útil; ese número acota el alcance real.
- Menciones de tatuaje/cicatriz dentro de síntesis y condiciones de ingreso (el 05a solo encontró 12 con `ILIKE '%TATUAJE%'`).

### P1 — Extracción estructurada de necropsias (`11_extraccion_necropsias_llm`)
- Reutiliza los modelos Pydantic del notebook 01 (`Tatuaje`, `SenaParticular`, `Prenda` y la parte de `MediaFiliacion` que aplique), con entrada de **solo texto**.
- Entrada por registro: ropa + pertenencias + condiciones de ingreso + descripción de tatuajes, cicatrices y vestimenta (si existen). Síntesis de los hechos solo como fuente de rasgos, nunca copiada.
- Mismas reglas: `SIN_DATO` si no se menciona, nunca nombres, normalización con `SINONIMOS`.
- Edad textual ("40-50 AÑOS") → `edad_min` / `edad_max` numéricos.
- Salidas: `necropsias_extraccion_bronze`, `necropsias_persona_silver`, `necropsias_{tatuajes,senas,vestimenta}_silver`, `necropsias_texto_embedding`, todas con la **misma función** de texto canónico que las fichas.
- Validación: muestra manual estratificada (el equipo revisa dentro de Databricks) con exactitud por campo.

### P2 — Pseudo-fichas y partición (`12_pseudofichas`)
- No hay pares reales ficha–necropsia. Igual que en el Avance 3: de cada necropsia **IDENTIFICADA** con ≥ 1 faceta se genera una pseudo-ficha en **estilo ficha** (vocabulario de las fichas), con ruido en tres niveles:
  - omisión de facetas (familia que no recuerda un tatuaje),
  - sinónimos y colores cercanos (azul marino ↔ azul),
  - error de edad (±3/±7/±12 años) y de región corporal (antebrazo ↔ brazo),
  - prendas cambiadas (la ropa de la desaparición no siempre es la del hallazgo).
- Galería de evaluación = 60 fichas reales + todas las pseudo-fichas (distractores reales incluidos).
- Partición **agrupada por necropsia** (entrenamiento / validación / prueba); la prueba no se toca hasta P7.

### P3 — Conjunto de candidatos (`13_candidatos_blocking`)
Una sola tabla de pares que usan **todos** los modelos:
- Duro: mismo sexo (si ambos lo tienen) y `fecha_hechos ≤ fecha_ingreso`.
- Suave: |Δedad| ≤ 15 contra el rango `edad_min–edad_max`, solo si ambos datos existen.
- Se reporta *pair completeness* (fracción de pares verdaderos que sobreviven) y reducción del espacio.

### P4 — Vector de comparación por faceta (`14_features_comparacion`)
| Faceta | Señales |
|---|---|
| Tatuajes | Emparejamiento uno a uno entre conjuntos (motivo, región, lateralidad, color); coseno del texto de la faceta; nº de tatuajes en ambos lados |
| Vestimenta | Emparejamiento entre prendas (tipo + color + marca); coseno de la faceta |
| Señas / cicatrices | Coincidencia de tipo y región; coseno de la faceta |
| Rasgos | Edad dentro del rango, Δestatura, complexión, cabello (si existen) |
| Contexto | Días entre hechos e ingreso; mismo estado/municipio |
| Disponibilidad | Indicadores de faceta presente en ficha / en necropsia por separado (lección del Avance 3) |

### P5 — Seis modelos individuales, mismo candidato, mismo MLflow
| # | Familia | Modelo | Usa P4 |
|---|---|---|---|
| 1 | Léxica | BM25 por faceta, suma ponderada | No |
| 2 | Probabilística no supervisada | Fellegi-Sunter con EM (Splink) | Sí (niveles discretos) |
| 3 | Supervisada lineal | Regresión logística (B2 reentrenado con facetas nuevas) | Sí |
| 4 | Learning-to-rank | LambdaMART (LightGBM `lambdarank`), grupos = necropsia | Sí |
| 5 | Semántica densa | Embeddings multilingües por faceta + k-NN en Vector Search (índices `vs_*_idx`) | No |
| 6 | Semántica de interacción | Cross-encoder multilingüe re-rankeando el top-50 | No |

- Una corrida MLflow por modelo y nivel de ruido: parámetros, Recall@k, MRR, IC 95 % por bootstrap sobre consultas, curva Recall@k. Nada de filas ni textos.
- Los pesos de facetas (0.65/0.25/0.10 del notebook 06) dejan de fijarse a mano: son hiperparámetros del modelo 1 y 5, y se aprenden en 2–4.

### P6 — Ajuste de los 2 mejores
- Criterio de selección: Recall@10 en validación con ruido alto y luego MRR; desempate por interpretabilidad para el perito.
- Búsqueda de hiperparámetros (Optuna) registrada en MLflow, con validación cruzada agrupada.

### P7 — Modelo final y salida pericial
- Evaluación única en prueba: (A) pseudo-fichas por nivel de ruido; (B) fichas reales con expediente conocido, si el equipo logra identificarlas; (C) top-k para necropsias NO IDENTIFICADAS, revisión cualitativa con el perito.
- Salida: tabla `candidatos_top_k` (id_necropsia seudonimizado, id_ficha, score, facetas que coinciden).
- Meta: Recall@10 ≥ 0.97 (≥ 0.80 con ruido alto).

## 3.1 Cobertura de la rúbrica

| Criterio (pts) | Cómo se cubre |
|---|---|
| **Comparativa (60)**: ≥ 6 modelos individuales con algoritmos variados, ordenados por la métrica principal, ≥ 2 métricas más y tiempos de entrenamiento | 7 modelos + baseline SQL + azar. `tabla_comparativa()` ordena por Recall@10 y muestra MRR, Recall@1/5/20, PR-AUC, Recall@10 con ruido alto, tiempo de entrenamiento y de inferencia |
| **Ajuste fino (30)**: hiperparámetros con suficientes iteraciones | Optuna sobre los 2 mejores, ≥ 50 trials cada uno, un run hijo por trial bajo `ajuste_top2`, validación cruzada agrupada; se registra la curva de optimización y la importancia de hiperparámetros |
| **Modelo individual (10)**: argumentos sólidos, trade-offs, interpretabilidad | Comparar los 2 ajustados en Recall@10, robustez al ruido alto, latencia y explicabilidad por faceta para el perito; la justificación se escribe en el notebook final y en la bitácora |

Arnés: `Notebooks/Databricks/00_config_evaluacion.py` (subido a `/Users/<usuario>/proyecto_mna/avance4/`; autoprueba sintética OK).

## 4. Riesgos

1. **Pocas necropsias con facetas.** Si P0 muestra que casi nadie tiene tatuajes o ropa descrita, el techo de cualquier modelo es bajo; se reporta como hallazgo y como recomendación de captura al SEMEFO.
2. **Galería de 60 fichas reales.** Recall@10 sobre 60 es optimista; siempre se reporta junto al azar (B0) con el mismo tamaño de galería.
3. **Pseudo-fichas = cota optimista.** El ruido se diseña desde las variantes reales observadas en el EDA de fichas.
4. **Fuga.** Las pseudo-fichas no se generan desde el mismo texto canónico que se compara; se generan desde la extracción estructurada con ruido aplicado antes de renderizar.

## 5. Preguntas abiertas para el equipo
- ¿Existe alguna lista de fichas reales con expediente conocido para la evaluación (B)?
- ¿El perito puede validar los umbrales de edad y la ventana temporal?
- ¿Se usa el mismo endpoint `gema` para la extracción de necropsias o un modelo de texto más barato?
