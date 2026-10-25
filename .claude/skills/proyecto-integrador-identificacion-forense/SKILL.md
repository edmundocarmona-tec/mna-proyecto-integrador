---
name: "proyecto-integrador-identificacion-forense"
description: "Usar como contexto cuando se trabaje en el Proyecto Integrador MNA de Edmundo sobre cotejo de personas desaparecidas con registros forenses de Querétaro (ante-mortem/post-mortem, Databricks Vector Search, CRISP-ML)."
---

# Proyecto Integrador MNA — Identificación cruzada de personas desaparecidas (Querétaro)

## Resumen del proyecto

Proyecto Integrador de la Maestría en Analítica de Negocios (MNA) de Edmundo Carmona Galindo (Solutions Architect en Databricks, Querétaro). Propone un proceso de **cotejo analítico** entre:

- **Fichas de búsqueda de personas desaparecidas** (datos ante-mortem) — de la Comisión Local de Búsqueda de Personas del estado de Querétaro y la Comisión Nacional de Búsqueda (CNB).
- **Banco de datos forense de cadáveres no identificados** (datos post-mortem) — el "Libro de Registro de Necropsias y Reconocimientos 2026", base institucional real del Centro de Resguardo Forense e Identificación Humana de la Fiscalía General del Estado de Querétaro (SEMEFO), ya obtenida por Edmundo: 1,100 registros, 49 campos; 568 identificados / 163 no identificados.

Colabora con la **Dirección de Servicios Periciales de la Fiscalía General del Estado de Querétaro**.

## Alcance y enfoque técnico (decisiones ya tomadas — no reabrir sin que el usuario lo pida)

- Enfoque en **análisis de datos**, no en rediseñar el proceso institucional de búsqueda/identificación.
- **Técnica de ML: recuperación por similitud (retrieval), NO clasificación supervisada.** Se descartó la clasificación supervisada porque no hay pares etiquetados confiables match/no-match ni conteo confiable de positivos — es un problema de record linkage con clase positiva casi inexistente, no clasificación binaria balanceada.
- **Pipeline propuesto:**
  1. Extracción: cada foto en vida de una ficha de búsqueda se procesa con un LLM multimodal usando un esquema fijo de salida (tatuajes, cicatrices, rasgos, complexión) — no texto libre, para reducir ruido en el embedding.
  2. Vectorización: se embeben las descripciones estructuradas + campos ya estructurados de ambos lados.
  3. Índice y búsqueda: **Databricks Vector Search** (decisión explícita de Edmundo, alineada con su stack profesional) indexa los registros del libro de necropsias; para cada ficha se hace k-NN con filtrado previo por reglas duras (sexo, edad, fechas, municipio — "blocking").
  4. Salida: ranking de candidatos por score + justificación textual, para revisión pericial (el modelo sugiere, el perito confirma — nunca identifica por sí solo).
- **Cotejo NO supervisado (restricción vigente desde el Avance 4):** no existen pares ficha ↔ necropsia confirmados; las tablas de Databricks no traen nombres que permitan cruzarlos. No se entrenan modelos supervisados, no se generan pseudo-fichas ni datos sintéticos, y no se vuelven a extraer fichas o necropsias con LLM: se usan las tablas reales existentes (`fichas_vector_search_gold`, 75 fichas; `necropsias_enriquecidas`, 1,246 registros). Recall@k y MRR solo aplican si el equipo consigue pares confirmados; mientras tanto la evaluación usa métricas sin etiquetas y revisión pericial del top-k. (La idea original de usar los identificados del libro como pares por coincidencia de nombre queda descartada.)
- **Fase 1 (alcance de este proyecto integrador) se limita a cotejo por descripciones TEXTUALES.** El matching visual foto-a-foto o superposición craneofacial (tipo Skeleton-ID/Face2Skull) queda fuera de alcance y se documenta como trabajo futuro — el libro sí tiene fotos reales del cuerpo, pero procesarlas queda para una fase posterior.

## Estructura de los datos (ya verificada, no volver a inferir)

**Ficha de búsqueda (ante-mortem):** folio único, carpeta de investigación, edad/sexo/nacionalidad al momento de la desaparición, complexión y rasgos físicos detallados (cara, color de piel, cabello, ojos, nariz, boca, labios, estatura, peso), señas particulares en texto libre, prendas de vestir, fecha/hora/lugar de los hechos, fotografía en vida.

**Libro de Registro de Necropsias y Reconocimientos 2026 (post-mortem):** 1,100 registros, 49 campos en 6 bloques — datos de necropsia (fechas, perito, carpeta de investigación, cadena de custodia), datos del lugar (municipio, lugar, domicilio de residencia), datos del fallecido (nombre, edad, sexo, estado civil, ocupación), datos del cuerpo (tipo de indicio, condiciones de ingreso, estatus identificado/no identificado, vestimenta, cicatrices, tatuajes, señas de reconocimiento), resultados SEMEFO (causa/tipo de muerte, cronodiagnóstico, reseña del hecho, estudios de química/genética/balística/patología), y entrega/estatus (gaveta, fecha y responsable de entrega, días hasta entrega o inhumación). No tiene datos dentales/dactiloscópicos/genéticos estructurados como campos propios, solo indica si se tomó "MUESTRAS".

Tabla comparativa de variables homologables entre ambos lados ya está definida en el documento del protocolo (ver sección "Entendimiento de los datos").

## Referencias bibliográficas verificadas (APA 7, ya confirmadas por búsqueda web, usar tal cual)

- Databricks. (s. f.). *Vector Search*. Databricks Documentation. https://docs.databricks.com/aws/en/ai-search/ai-search
- Emisoras Unidas. (2026, 2 de julio). *Inacif identifica por primera vez a una víctima mediante herramienta de inteligencia artificial*. https://emisorasunidas.com/nacional/2026/07/02/inacif-identifica-victima-herramienta-inteligencia-artificial
- Fellegi, I. P., & Sunter, A. B. (1969). A theory for record linkage. *Journal of the American Statistical Association, 64*(328), 1183–1210. https://doi.org/10.1080/01621459.1969.10501049
- Instituto Mexicano de Derechos Humanos y Democracia. (2025). *La desaparición de personas en 2025*. https://www.imdhd.org/comunicacion/blog/la-desaparicion-de-personas-en-2025/
- INTERPOL. (2023). *Disaster victim identification guide*. https://www.interpol.int/content/download/589/file/DVI_DVI%20Guide%202023.pdf
- Mireles Loera, O. (2022). Realce de tatuajes como método de identificación forense. *Revista Mexicana de Medicina Forense y Ciencias de la Salud, 7*(2). https://revmedforense.uv.mx/index.php/RevINMEFO/article/view/2978
- Secretaría de Gobernación, Comisión Nacional de Búsqueda. (s. f.). *Registro Nacional de Personas Desaparecidas y No Localizadas: Versión pública de estadísticas* [Panel de datos]. https://versionpublicarnpdno.segob.gob.mx/
- Skeleton-ID. (s. f.). *Superposición craneofacial*. https://skeleton-id.com/craniofacial-superimposition-skeleton-id-software/
- Studer, S., Bui, T. B., Drescher, C., Hanuschkin, A., Winkler, L., Peters, S., & Müller, K.-R. (2021). Towards CRISP-ML(Q): A machine learning process model with quality assurance methodology. *Machine Learning and Knowledge Extraction, 3*(2), 392–413. https://doi.org/10.3390/make3020020

**Corrección importante ya hecha:** "Angelus" NO es un precedente de reconocimiento de tatuajes — es un sistema de la UNAM/CIDE para archivos de la Guerra Sucia (1964-1985), sin reconocimiento de imágenes. El precedente correcto de cotejo de tatuajes por IA es **IdentIA**, de **LAB-CO** (no es un laboratorio del CIDE, es una organización aparte), desplegado en el INACIF de Guatemala. No reintroducir la referencia errónea a Angelus como precedente de tatuajes.

## Estructura del documento (protocolo de investigación, formato CRISP-ML)

1. Antecedentes
2. Entendimiento del negocio (formulación del problema, contexto, objetivos, preguntas clave, involucrados)
3. Entendimiento de los datos (descripción, técnica de ML, variables de entrada/salida)
4. Convenios (pendiente: convenio de privacidad con la Fiscalía, dado el acceso a datos reales sensibles)
5. Portada, índice, bibliografía en APA 7, anexos

## Alertas de sensibilidad de datos (mantener presentes en cualquier entrega)

- Las fichas de búsqueda de ejemplo y el libro de necropsias contienen **identidades reales** de personas (nombres, fotos, folios). No pegar filas/capturas reales con esos datos visibles en entregables que se vayan a compartir o publicar; usar tablas de estructura genérica en su lugar.
- Dado el acceso a datos reales sensibles (causas de muerte, menores de edad en los registros), el convenio de confidencialidad con la Fiscalía es indispensable para la entrega formal, no opcional.

## Nota de alcance distinto — NO confundir con este proyecto

Edmundo también cursa el curso **TC5038** (caso Marriott, Design Thinking + Scrum, proyecto de equipo), que es un curso completamente distinto, sin relación con este Proyecto Integrador MNA. Si en una sesión aparece contexto de Marriott/TC5038/Design Thinking, **no mezclarlo** con el proyecto de identificación forense.