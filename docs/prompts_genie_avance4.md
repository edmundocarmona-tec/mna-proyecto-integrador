# Prompts para Genie Code — Avance 4

> **AVISO (Decisión 24): el Avance 4 es un cotejo NO supervisado.** No hay pares ficha ↔ necropsia confirmados.
> No se entrenan modelos supervisados, no se generan pseudo-fichas ni datos sintéticos, y no se vuelve a extraer
> nada con LLM. Se compara directamente `necropsias_enriquecidas` (consultas) contra `fichas_vector_search_gold`
> (galería, 75 fichas). Las secciones de este documento sobre extracción de necropsias (P1), pseudo-fichas (P2),
> modelos supervisados (regresión logística, LambdaMART) y Recall@k quedan **descartadas**; ver `CLAUDE.md`.


Redactados por Claude Code (24 oct 2026) para ejecutar en Databricks con datos reales. Uso:

1. Crear el notebook vacío en `/Users/<usuario>/proyecto_mna/avance4/` (junto a `00_config_evaluacion`).
2. Abrir Genie Code en ese notebook, pegar **el bloque de contexto común + el prompt del paso**.
3. Revisar el código antes de correrlo y verificar que solo imprime agregados.
4. Pasar al siguiente paso solo cuando el anterior haya dejado sus tablas.
5. Registrar en `docs/declaracion_uso_ia.md` qué generó Genie y qué se corrigió.

Orden: **11 → 12 → 13 → 20**. Los modelos 21–27 siguen la plantilla del paso 20.

---

## Bloque de contexto común (pegar siempre primero)

```
CONTEXTO DEL PROYECTO (no lo cambies):
Cotejo forense: para cada registro de necropsia (consulta) se rankean fichas de búsqueda de personas
desaparecidas (galería). Catálogo/esquema: proyecto_mna.fichas_busqueda.

REGLAS OBLIGATORIAS:
1. La primera celda de código es:  %run ./00_config_evaluacion
   Usa sus constantes (T_NECROPSIAS, T_NEC_VESTIMENTA, T_PSEUDOFICHAS, T_VERDAD, T_CANDIDATOS, etc.)
   y sus funciones (corrida, fase_padre, evaluar, tabla_comparativa). NO modifiques ese notebook
   y NO crees otro experimento de MLflow.
2. Privacidad: los datos son de personas reales. NUNCA uses display(), show(), print() ni head()
   sobre filas o textos. Solo imprime conteos, porcentajes, distribuciones agregadas y esquemas.
   Nunca extraigas ni guardes nombres, folios ni carpetas de investigación.
3. MLflow: solo parámetros, métricas y gráficas de agregados. Prohibido mlflow.log_input,
   log_table, log_text con datos, y autolog con datasets. No actives tracing de LLM.
4. Escribe en Unity Catalog con saveAsTable en el mismo esquema. Si la tabla existe, sobrescríbela
   solo si el widget "sobrescribir" = "true".
5. Comenta el código en español, con celdas markdown que expliquen cada paso.
```

---

## Paso 11 — `11_extraccion_necropsias_llm`

```
OBJETIVO: llevar las necropsias al MISMO esquema estructurado que ya tienen las fichas, para poder
comparar prenda contra prenda y tatuaje contra tatuaje.

ENTRADA: T_NECROPSIAS (necropsias_enriquecidas). Columnas útiles: no_nec, anio, fecha_ingreso,
tipo_ingreso, sexo_norm, status, ropa, pertenencias, descr_tatuajes, tiene_tatuaje_bin,
tiene_cicatriz, descr_vestimenta, edad, municipio_hechos.
NO uses las columnas sintesis, resena_hecho ni causa_muerte (pueden contener nombres).

ESQUEMA: copia de /Users/<usuario>/.../01_extraccion_fichas_llm los modelos Pydantic Prenda,
Tatuaje y SenaParticular, el diccionario SINONIMOS y la función normalizar_salida, sin cambiar
sus vocabularios. Agrega un modelo nuevo Pertenencia con tipo Literal["DOCUMENTO","CELULAR",
"JOYERIA","RELOJ","CARTERA","MOCHILA","DINERO","LLAVES","OTRA","SIN_DATO"], descripcion y marca.
El modelo de salida NecropsiaExtraida tiene listas: prendas, tatuajes, senas, pertenencias.

LLM: cliente OpenAI-compatible como en el notebook 01, endpoint en un widget (por defecto
proyecto_mna.fichas_busqueda.gema), entrada SOLO TEXTO: ropa + pertenencias + descr_tatuajes +
descr_vestimenta + tiene_cicatriz, etiquetadas. Prompt: extraer solo lo mencionado, SIN_DATO si no
se menciona, nunca inventar, nunca transcribir nombres. Validar con Pydantic, 2 reintentos.
Widget "limite" (por defecto 20) para probar antes de correr todo; widget "max_workers" = 4.

EDAD: no la pidas al LLM. Convierte la columna edad (texto tipo "40-50 AÑOS", "35 AÑOS") a
edad_min y edad_max enteros con regex; null si no se puede.

IDEMPOTENCIA: llave = sha256(texto de entrada) + modelo + huella del prompt (sha256 del prompt
y del JSON schema). No reprocesar filas que ya están en bronze con la misma llave.

SALIDAS (constantes de 00):
- T_NEC_BRONZE: no_nec, llave, json crudo, ok, error, ts
- T_NEC_VESTIMENTA, T_NEC_TATUAJES, T_NEC_PERTENENCIAS: una fila por elemento, con no_nec y n
- T_NEC_PERSONA: no_nec, sexo_norm, edad_min, edad_max, fecha_ingreso, tipo_ingreso, status,
  n_prendas, n_tatuajes, n_senas, n_pertenencias
- T_NEC_TEXTO: no_nec + texto_vestimenta, texto_tatuajes, texto_senas, texto_pertenencias,
  construidos con LA MISMA función de texto canónico del notebook 01 (cópiala tal cual).

CONTROL DE CALIDAD (solo agregados): filas procesadas, % OK, errores por tipo, % con ≥1 prenda,
distribución de tipo y color de prenda (conteos), % con edad convertida.
Registra en MLflow un run con corrida("extraccion_necropsias", familia="datos", fase="datos")
con esos agregados como métricas y el endpoint y la huella del prompt como parámetros.
```

---

## Paso 12 — `12_pseudofichas`

```
OBJETIVO: crear la verdad de evaluación. No hay pares reales ficha–necropsia, así que de cada
necropsia IDENTIFICADA se genera una pseudo-ficha (lo que una familia habría reportado), con ruido.

BASE: necropsias con status 'IDENTIFICADO NO RECLAMADO', tipo_ingreso distinto de FETO y al menos
una prenda en T_NEC_VESTIMENTA (se esperan ~250). Imprime solo cuántas son.

GENERACIÓN: parte de las tablas silver estructuradas (NO del texto canónico, para evitar fuga).
Para cada necropsia crea TRES pseudo-fichas, una por nivel de ruido (bajo, medio, alto):
- Omitir cada prenda con prob. 0.10 / 0.25 / 0.40
- Cambiar color por uno cercano (diccionario: AZUL↔AZUL_MARINO, NEGRO↔GRIS, BLANCO↔BEIGE,
  CAFE↔BEIGE, ROJO↔VINO, VERDE↔VERDE_OLIVO; ajusta a los valores reales del vocabulario) con
  prob. 0.10 / 0.20 / 0.35
- Cambiar el tipo de prenda por uno parecido (PLAYERA↔CAMISA, SUDADERA↔CHAMARRA, TENIS↔CALZADO)
  con prob. 0.05 / 0.15 / 0.25
- Borrar la marca con prob. 0.30 / 0.50 / 0.70
- Tatuajes: omitir con prob. 0.10 / 0.25 / 0.40; región adyacente (ANTEBRAZO↔BRAZO,
  MANO↔DEDOS, PECHO↔ABDOMEN) con prob. 0.10 / 0.20 / 0.30
- Edad (si existe): punto medio del rango + ruido normal con desviación 3 / 6 / 10 años
- fecha_hechos = fecha_ingreso − U(1, 30) / U(1, 90) / U(1, 365) días
Semilla fija (SEMILLA de 00). Las probabilidades van en un diccionario al inicio y se registran
como parámetros de MLflow.

SALIDAS:
- T_PSEUDOFICHAS: id_ficha ("PS_" + sha256(no_nec|nivel)[:12]), nivel_ruido, sexo, edad,
  fecha_hechos y texto_vestimenta/texto_tatuajes/texto_senas/texto_pertenencias renderizados con
  la MISMA función de texto canónico que las fichas reales. También tablas hijas con las prendas
  y tatuajes resultantes (mismas columnas que las silver de fichas).
- T_VERDAD: id_consulta = no_nec || '|' || nivel_ruido, id_ficha, nivel_ruido, split.
  Split agrupado por no_nec (las 3 copias de una necropsia caen en el mismo split):
  60 % train, 20 % val, 20 % test, por hash estable de no_nec.

Registra un run corrida("pseudofichas", familia="datos", fase="datos") con conteos por nivel y
split y las probabilidades de ruido como parámetros. Solo agregados.
```

---

## Paso 13 — `13_candidatos_blocking`

```
OBJETIVO: una sola tabla de pares candidatos que usarán TODOS los modelos.

CONSULTAS:
(a) evaluación: cada id_consulta de T_VERDAD (necropsia|nivel);
(b) operación: necropsias NO IDENTIFICADO con ≥1 prenda o tatuaje, tipo_ingreso ≠ FETO
    (id_consulta = no_nec || '|real'); se usan solo para revisión cualitativa.
GALERÍA: las 60 fichas reales (T_FICHAS_TEXTO) + las pseudo-fichas. Una consulta de evaluación con
nivel X solo ve las fichas reales + las pseudo-fichas de nivel X (para que las copias de otros
niveles no compitan). Las consultas reales solo ven las fichas reales.

REGLAS DE BLOCKING (parámetros en un diccionario al inicio):
- Sexo igual cuando ambos lados lo tienen (INDETERMINADO o nulo pasa).
- fecha_hechos ≤ fecha_ingreso cuando ambos existen (convertir fechas de texto con to_date y
  varios formatos; contar cuántas no se pudieron convertir).
- |edad ficha − rango de edad necropsia| ≤ 15 años, solo si ambos existen.

SALIDA: T_CANDIDATOS con id_consulta, id_ficha, nivel_ruido, split, tipo_consulta
('evaluacion'/'real').

MÉTRICAS (solo agregados), por nivel de ruido y por split:
pair completeness (fracción de pares verdaderos de T_VERDAD que sobreviven), candidatos promedio
por consulta, reducción del espacio vs. producto cruzado, cuántas consultas reales quedan sin
candidatos. Registra un run corrida("blocking", familia="datos", fase="datos").
```

---

## Paso 20 — `20_baselines_azar_sql` (plantilla para todos los modelos)

```
OBJETIVO: los dos baselines del Avance 4, registrados con el arnés de 00.

DATOS: T_CANDIDATOS filtrado a tipo_consulta='evaluacion'; T_VERDAD. Para cada par, une las
tablas hijas de prendas/tatuajes de la necropsia (T_NEC_*) y de la ficha (silver de fichas reales
o tablas hijas de pseudo-fichas).

MODELO B0 — azar: score = número aleatorio con SEMILLA.

MODELO 0 — reglas SQL (todo en Spark SQL):
score = suma ponderada de coincidencias exactas por faceta:
- prendas: número de prendas de la necropsia con una prenda de la ficha del mismo tipo Y color
  (cada prenda de la ficha se usa una sola vez), +1 extra si además coincide la marca
- tatuajes: número de tatuajes con misma región Y misma categoría o motivo
- edad: +1 si la edad de la ficha cae dentro del rango de la necropsia
Pesos en un diccionario (por defecto 1.0) registrados como parámetros.

REGISTRO: dentro de with fase_padre("comparacion"):, una corrida por modelo:
  with corrida("0_sql_reglas", familia="deterministico", fase="comparacion", params=PESOS) as c:
      with c.entrenamiento(): pass          # no hay entrenamiento
      with c.inferencia(): rankings = ...   # pandas: id_consulta, id_ficha, score
      c.registrar(rankings, verdad)
Evalúa en el split 'test' (y también en 'val' en corridas aparte con tag split=val).
Repite la evaluación por nivel de ruido con nivel_ruido="bajo"/"medio"/"alto" filtrando la verdad.
Al final muestra tabla_comparativa() y grafica_comparativa() (solo agregados).
```

Para los modelos 1–7 se reutiliza este prompt cambiando la sección MODELO y la familia:
`1_fuzzy` (cadenas_aproximadas), `2_bm25` (lexico), `3_splink_em` (probabilistico),
`4_regresion_logistica` (supervisada_lineal), `5_lambdamart` (learning_to_rank),
`6_embeddings_vs` (semantica_densa), `7_cross_encoder` (semantica_interaccion).
Los supervisados (4, 5) entrenan en 'train', eligen hiperparámetros en 'val' y se evalúan en 'test'.
