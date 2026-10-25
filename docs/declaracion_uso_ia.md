# Declaración de uso de inteligencia artificial

**Proyecto Integrador MNA — Equipo 23, Grupo 10**
Cotejo de fichas de búsqueda de personas desaparecidas (ante-mortem) con registros forenses del SEMEFO de Querétaro (post-mortem)
Autor de esta declaración: Edmundo Carmona Galindo
Última actualización: 25 de octubre de 2026 (documento vivo; se amplía en cada avance)

---

## 1. Herramientas de IA utilizadas

| Herramienta | Uso | Tipo de uso |
|---|---|---|
| **Claude (Anthropic)**, en un Proyecto de claude.ai | Investigación de fuentes, diseño metodológico, generación y depuración de código, auditoría de notebooks, redacción y revisión de textos | Asistente de trabajo |
| **Claude Code (Anthropic)**, CLI sobre el repositorio local y el CLI de Databricks | Revisión del repositorio, planeación del Avance 4, perfilado de tablas con consultas de agregados, redacción de esta declaración | Asistente de trabajo |
| **Databricks Genie Code** | Ejecución asistida del EDA de las fichas sobre las tablas en Databricks | Asistente de trabajo |
| **LLM multimodal servido en Databricks AI Gateway** (endpoint `proyecto_mna.fichas_busqueda.gema`) | Extracción estructurada de rasgos a partir de las fichas de búsqueda | **Componente del sistema propuesto** (no asistente de redacción) |

## 2. División de responsabilidades

- **La IA propuso, explicó y redactó.** Propuso alternativas metodológicas, escribió borradores de código y de texto, y auditó el trabajo existente.
- **El equipo decidió, ejecutó y validó.**
  - Las decisiones de alcance las tomó el equipo: enfoque, herramientas, uso de datos y qué se entrega.
  - Las ejecuciones con datos reales se hicieron en el entorno del equipo, salvo lo indicado en la sección 6.
  - Los resultados de la IA se revisaron contra los datos. En varios casos el autor detectó errores en las salidas y pidió corregirlos (ver Decisiones 9, 10 y 13).
- **Ninguna cifra reportada proviene de la IA sin ejecución.** Todas las métricas salen de notebooks ejecutados sobre los datos.

---

## 3. Decisiones — Planteamiento del proyecto

Decisión 1: Plantear el problema como recuperación por similitud (vinculación de registros) y no como clasificación supervisada
- Justificación: No existen pares ficha–expediente etiquetados como coincidencia o no coincidencia, y la clase positiva es casi inexistente. El perito necesita una lista ordenada de candidatos que revisar, no una etiqueta binaria.
- Alternativa descartada y motivo: Entrenar un clasificador binario de "identificado / no identificado". No responde la pregunta de negocio (¿qué expediente corresponde a esta ficha?) y no hay etiquetas confiables para entrenarlo.
- Participación de la IA: Claude expuso las ventajas y limitaciones de ambos enfoques. La decisión la tomó el autor.

Decisión 2: Usar Databricks Vector Search como motor de búsqueda vectorial
- Justificación: Se integra con Unity Catalog, que da gobierno y control de acceso sobre datos sensibles, y con MLflow. Además es el stack que el autor domina profesionalmente.
- Alternativa descartada y motivo: Una base vectorial externa (servicio en la nube de terceros), porque implicaría sacar datos sensibles del entorno gobernado.
- Participación de la IA: Ninguna en la elección; fue decisión explícita del autor.

Decisión 3: Limitar la Fase 1 al cotejo de descripciones textuales
- Justificación: El cotejo visual (foto contra foto, superposición craneofacial) requiere otro tipo de datos, validación pericial y tiempos que exceden el curso.
- Alternativa descartada y motivo: Incluir matching de imágenes desde el inicio. Se documenta como trabajo futuro.
- Participación de la IA: Claude investigó precedentes (IdentIA/INACIF, Skeleton-ID). Se corrigió una referencia errónea que atribuía reconocimiento de tatuajes al sistema Angelus.

Decisión 4: Validar con registros ya identificados como proxy de pares positivos
- Justificación: Permite medir recall@k sin depender de que la Fiscalía etiquete pares manualmente.
- Alternativa descartada y motivo: Esperar un conjunto etiquetado por peritos, que no está disponible en los tiempos del curso.
- Participación de la IA: Claude propuso el esquema de validación. El equipo lo aprobó.

## 4. Decisiones — Datos ante-mortem y EDA (Avance 2)

Decisión 5: Usar las fichas públicas de la Alerta Nacional de Búsqueda como fuente de fichas
- Justificación: No existe una base oficial descargable. Desde el protocolo publicado en el DOF (febrero de 2026), la ficha tiene un formato nacional homologado y la difunden instituciones privadas en PDF con texto extraíble.
- Alternativa descartada y motivo: Hacer scraping de portales sin formato homogéneo, o esperar las fichas de la Comisión Local de Búsqueda, que requieren convenio.
- Participación de la IA: Claude realizó la búsqueda de fuentes públicas. El autor validó y recolectó las fichas.

Decisión 6: Extraer los rasgos con un LLM multimodal y un esquema fijo de salida (Pydantic con vocabularios controlados)
- Justificación: El texto libre introduce ruido en los embeddings. Un esquema fijo, con `SIN_DATO` explícito para ausencias, hace comparables las fichas entre sí y con el libro de necropsias.
- Alternativa descartada y motivo: Pedir al LLM una descripción libre, porque genera variabilidad de redacción y posibles alucinaciones difíciles de auditar.
- Participación de la IA: Claude escribió el notebook de extracción (`01_extraccion_fichas_llm.py`) y lo adaptó a la Responses API del AI Gateway. El autor lo ejecutó en Databricks.

Decisión 7: Excluir de la extracción con LLM el perfil genético y las huellas dactilares
- Justificación: Son datos de laboratorio con formato propio. Deben manejarse como tablas separadas, no inferirse de una imagen.
- Alternativa descartada y motivo: Extraerlos junto con los rasgos físicos, con riesgo de errores sobre evidencia crítica.
- Participación de la IA: Claude la propuso a partir de las cinco señales de identificación descritas por el perito colaborador.

Decisión 8: Agregar una capa de normalización de sinónimos con respaldo auditable, en lugar de rechazar la salida del modelo
- Justificación: El modelo devolvía sinónimos válidos ("LISO" por "LACIO", "MEJILLA" por "CARA") que rompían la validación. Normalizarlos y registrar cada ajuste en `notas_extraccion` conserva la información y deja rastro.
- Alternativa descartada y motivo: Descartar los registros inválidos, con pérdida de datos, o aflojar el esquema, con pérdida de comparabilidad.
- Participación de la IA: El autor detectó los errores al ejecutar. Claude implementó la normalización.

Decisión 9: Tomar la vestimenta solo del texto de la ficha (no de la foto) y registrar alteraciones como "nariz operada" como seña de tipo cirugía
- Justificación: La foto puede ser anterior a la desaparición, así que la ropa que aparece en ella no describe los hechos. Además, una alteración quirúrgica es una seña identificatoria.
- Alternativa descartada y motivo: Mantener el prompt original, que tomaba la ropa de la imagen y perdía la cirugía como `SIN_DATO`.
- Participación de la IA: **El autor detectó ambos errores** al revisar una ficha real. Claude corrigió las reglas del prompt.

Decisión 10: Adaptar el EDA a un problema sin variable objetivo y no imputar rasgos faltantes
- Justificación:
  - Las preguntas estándar de correlación con el objetivo se reemplazaron por medidas de poder discriminante (entropía por rasgo).
  - En contexto forense, imputar un rasgo equivale a fabricar evidencia.
  - Los atípicos de estatura se marcan para revisión, no se eliminan, porque un rasgo poco común es justamente el que distingue a una persona.
- Alternativa descartada y motivo: Aplicar el EDA estándar con imputación y eliminación de atípicos, que contradice el objetivo forense.
- Participación de la IA: Claude redactó las instrucciones del EDA para Genie Code y el documento de resultados. En la revisión se corrigieron imprecisiones del borrador: la recomendación de imputar, una interpretación invertida de la asimetría, el uso incorrecto de "MNAR" y el porcentaje de tatuajes al 100 % (efecto de un error de extracción). El autor cuestionó la explicación de los atípicos por grupo de edad, y la redacción final se ajustó para distinguir artefactos de muestra de atípicos reales.

Decisión 11: Reportar solo agregados y seudonimizar los datos
- Justificación: Las fichas y el libro de necropsias contienen identidades reales. Ningún entregable debe mostrar nombres, folios, carpetas de investigación ni fotos.
- Alternativa descartada y motivo: Incluir ejemplos reales para ilustrar, por riesgo para las víctimas y sus familias, y porque no existe todavía un convenio de confidencialidad firmado.
- Participación de la IA: Claude aplicó la regla en todos los entregables y verificó con escaneos automáticos que no hubiera nombres ni carpetas en las salidas.

## 5. Decisiones — Modelo de referencia (Avance 3)

Decisión 12: Reconstruir el baseline como vinculación de registros, tras detectar fuga de información en la versión original
- Justificación: En el notebook original, el 76.5 % de las filas (4,070 de 5,317) estaban vacías y etiquetadas como "NO IDENTIFICADO". El AUC de 0.94 medía si la fila tenía datos, no algo forense. Además, el modelo respondía otra pregunta (identificado o no) en lugar de la de cotejo.
- Alternativa descartada y motivo: Entregar el modelo original con su AUC, que habría reportado un desempeño inexistente.
- Participación de la IA: Claude auditó el notebook contra la rúbrica y encontró la fuga. El equipo aceptó rehacer la sección.

Decisión 13: Validar con pseudo-fichas (copias con ruido controlado de registros identificados) y partición agrupada por ficha
- Justificación: No hay pares reales ficha–expediente. Las pseudo-fichas dan positivos conocidos y permiten probar la robustez con tres niveles de ruido. Agrupar por ficha evita que pares de una misma ficha caigan en entrenamiento y prueba.
- Alternativa descartada y motivo: Partición aleatoria por pares, que oculta el sobreajuste. Los resultados con pseudo-fichas se reportan como cota optimista.
- Participación de la IA: Claude diseñó el protocolo. Lo probó primero con datos simulados y después lo ejecutó con el registro real (ver sección 6). Durante la ejecución se encontró que una variable ("señas vacías") hacía que B2 rindiera menos que B1. Se separó en dos indicadores (lado ficha y lado expediente) y se ajustó con validación cruzada.

Decisión 14: Usar Recall@k y MRR como métricas principales
- Justificación: El perito revisa una lista corta. Lo que importa es si el expediente correcto aparece entre los primeros k. Recall@10 traduce directamente la reducción del espacio de revisión (de unos 1,247 expedientes a 10).
- Alternativa descartada y motivo: Exactitud o ROC-AUC a nivel de par, que son engañosas con clases extremadamente desbalanceadas. PR-AUC se reporta solo como complemento.
- Participación de la IA: Claude propuso las métricas y el umbral de negocio. El equipo los aprobó.

Decisión 15: Elegir como baseline oficial una regresión logística sobre vectores de comparación (B2), frente al azar (B0) y una heurística TF-IDF (B1)
- Justificación: Es la versión discriminativa del modelo clásico de Fellegi y Sunter. Es interpretable para el perito, tiene baja varianza con pocos positivos y no muestra sobreajuste (train ≈ CV ≈ 0.847). Obtuvo Recall@10 = 0.869, contra 0.028 del azar y 0.850 de la heurística.
- Alternativa descartada y motivo: Random Forest como baseline. Rindió menos en prueba (0.804) y memoriza con pocos positivos.
- Participación de la IA: Claude implementó y ejecutó la comparación. El diagnóstico de sobreajuste se corrigió para basarse en validación cruzada y no en una sola brecha de un conjunto de prueba pequeño.

Decisión 16: Aplicar un blocking amplio (mismo sexo y diferencia de edad de hasta 15 años, solo cuando ambos datos existen)
- Justificación: Reduce el cómputo y replica el primer filtro del perito, sin descartar candidatos por datos faltantes. Su costo se mide con *pair completeness*.
- Alternativa descartada y motivo: Tolerancias estrechas (por ejemplo, ±5 años), con riesgo de eliminar al candidato correcto por errores de estimación de edad post-mortem. Queda pendiente validarlas con el perito.
- Participación de la IA: Claude lo propuso. Los umbrales siguen abiertos a validación pericial.

## 6. Manejo de datos con herramientas de IA

- **Fichas de búsqueda.** Se procesaron dentro de Databricks: Volume de Unity Catalog y endpoint del AI Gateway. A la conversación con Claude se compartieron resúmenes agregados y una ficha de ejemplo para depurar el prompt de extracción.
- **Libro de necropsias.** Para cerrar el Avance 3, el archivo del registro se cargó en una sesión de Claude para ejecutar el notebook de punta a punta. Se verificó que las salidas del notebook no contuvieran nombres ni carpetas de investigación.
- **Entregables.** Contienen solo agregados y estructuras genéricas.
- **Claude Code (Avance 4).** Leyó el código fuente de los notebooks y los esquemas de las tablas. En Databricks ejecutó solo consultas de agregados (conteos, porcentajes de llenado y frecuencias de palabras con 40 o más apariciones) en el SQL warehouse del equipo. No vio registros individuales, nombres, folios ni carpetas. Las consultas se guardaron fuera del repositorio.

## 7. Decisiones — Modelos alternativos (Avance 4)

Decisión 17: Comparar tipos de búsqueda (familias de recuperación y puntuación) como los modelos alternativos del avance
- Justificación: El proyecto es de cotejo. La pregunta relevante es qué forma de buscar recupera mejor el expediente correcto: léxica, probabilística, supervisada, de ranking o semántica. Cada tipo es un algoritmo individual con sus propios hiperparámetros, y todos se evalúan sobre el mismo conjunto de candidatos.
- Alternativa descartada y motivo: Comparar solo clasificadores sobre el mismo vector de comparación, que dejaría sin probar la hipótesis central (que la representación semántica supera a la léxica).
- Participación de la IA: Idea del autor. Claude propuso las seis familias y el protocolo de comparación.

Decisión 18: Registrar todos los experimentos en MLflow (Databricks)
- Justificación: Reproducibilidad y comparación con criterios comunes (parámetros, métricas, versión de datos y semilla), y trazabilidad del ajuste de hiperparámetros, como pide CRISP-ML(Q).
- Alternativa descartada y motivo: Tablas de resultados manuales en el notebook, difíciles de auditar y de reproducir.
- Participación de la IA: Decisión del autor. Claude propone la estructura del experimento.

Decisión 19: Usar `necropsias_enriquecidas` como fuente consolidada post-mortem: la unión de `registro_necropsias_2020_2026` (1,246 registros) con `registro_necropsias_2026` (58 registros con detalle forense)
- Justificación: La tabla nueva amplía la cobertura (2006–2026) y la anterior aporta tatuajes, cicatrices, edad y municipio en los casos que comparten. La anterior es subconjunto completo de la nueva (`No = #`), lo que se verificó en el notebook 05a.
- Alternativa descartada y motivo: Usar solo una de las dos tablas. Con la nueva sola se pierde el detalle forense; con la anterior sola quedan apenas 58 casos.
- Participación de la IA: La consolidación la hizo el equipo (notebook 05a). Claude Code revisó el código y señaló que el detalle forense existe en solo 58 registros.

Decisión 20: Plantear la consulta en dirección necropsia → fichas: para cada persona registrada en el libro de necropsias se recupera un ranking de fichas de búsqueda candidatas
- Justificación: Es la pregunta operativa del SEMEFO ante un cuerpo sin identificar. Recall@k y MRR se miden por necropsia consultada.
- Alternativa descartada y motivo: Ficha → necropsias, como estaba en el planteamiento inicial. Sigue siendo válida para la Comisión de Búsqueda y se puede invertir con los mismos componentes, pero no es el uso que se evalúa en este avance.
- Participación de la IA: Decisión del autor. Claude Code detectó que los notebooks de Vector Search ya consultaban en esta dirección, mientras que la documentación decía lo contrario, y pidió fijarla antes de definir las métricas.

Decisión 21: Perfilar las tablas post-mortem antes de diseñar los modelos (fase P0), usando solo consultas de agregados
- Justificación: El alcance real del cotejo depende de qué facetas existen en ambos lados. Resultados principales (1,246 necropsias):
  - Ropa descrita en 847 registros (68 %). Es texto semiestructurado: prenda, color, marca, talla, leyendas.
  - Tatuajes en solo 12 registros y señas en 5, contra 46 y 17 de las 60 fichas.
  - Edad y municipio solo en 58 registros. Fecha de ingreso en prácticamente todos.
  - 892 no identificados y 354 identificados no reclamados; ningún identificado reclamado. 61 fetos y 93 restos óseos.
- Alternativa descartada y motivo: Diseñar los modelos asumiendo que los tatuajes, la faceta más discriminante de las fichas, estarían disponibles en las necropsias. El perfilado muestra que casi no se registran.
- Participación de la IA: Claude Code diseñó y ejecutó las consultas desde el CLI de Databricks con el perfil del autor. Solo obtuvo conteos y frecuencias de palabras de la ropa (las que aparecen 40 veces o más); no leyó registros individuales.

Decisión 22: Usar un cotejo determinístico tipo SQL (reglas de coincidencia exacta) como baseline y comparar contra él siete técnicas individuales: cadenas aproximadas (Jaro-Winkler/Levenshtein), BM25 por faceta, Fellegi-Sunter con EM (Splink), regresión logística, LambdaMART, embeddings densos con Vector Search y cross-encoder
- Justificación: Las reglas SQL reproducen el razonamiento del perito (misma prenda, mismo color, misma marca) y dan una referencia fácil de explicar. Las siete técnicas cubren de lo más rígido a lo más flexible y superan el mínimo de seis modelos individuales que pide la rúbrica. Varias comparten el emparejamiento elemento por elemento (asignación húngara entre prendas, tatuajes y pertenencias).
- Alternativa descartada y motivo: Contar las reglas SQL como uno de los seis modelos. Se prefirió usarlas como referencia, junto con el azar (B0), para que las técnicas de aprendizaje se midan contra el criterio de un perito.
- Participación de la IA: El autor propuso incluir el matching tipo SQL. Claude Code propuso el catálogo de técnicas, el orden y el emparejamiento por conjuntos. El autor lo aprobó.

Decisión 23: Evaluar todos los modelos con un arnés común (notebook `00_config_evaluacion`) y un solo experimento de MLflow (`/Shared/proyecto_mna/avance4_cotejo`), con runs padre por fase (comparación, ajuste de los 2 mejores, final)
- Justificación: La rúbrica pide ordenar por la métrica principal, mostrar al menos dos métricas más y reportar tiempos de entrenamiento. El arnés lo garantiza para todos los modelos por igual:
  - Cada modelo entrega un ranking con el mismo formato `(id_consulta, id_ficha, score)` y una sola función calcula Recall@1/5/10/20, MRR, PR-AUC a nivel par e IC 95 % por bootstrap.
  - Los empates se resuelven de forma pesimista, para no inflar a los modelos de reglas, que producen muchos scores iguales.
  - Si el blocking elimina la ficha correcta, cuenta como fallo, así su costo entra en la métrica final.
  - Se registran tiempos de entrenamiento e inferencia, y la versión Delta de cada tabla de entrada (reproducible sin subir datos).
  - Privacidad: autolog sin datasets ni ejemplos de entrada, tracing de LLM desactivado, solo gráficas de agregados. Los rankings van a una tabla de Unity Catalog con `run_id`, no a MLflow.
- Alternativa descartada y motivo: Que cada notebook calcule sus propias métricas, con riesgo de criterios distintos (manejo de empates, consultas excluidas) que harían injusta la comparación.
- Participación de la IA: Claude Code escribió el notebook, lo subió al workspace y lo ejecutó como job serverless con una autoprueba sobre datos sintéticos (ranking perfecto, al azar y todo empatado). La autoprueba pasó. No se usaron datos reales en esta prueba.

Decisión 24: Tratar el Avance 4 como un cotejo no supervisado: sin modelos supervisados, sin datos sintéticos y sin nuevas extracciones con LLM
- Justificación: No existen pares ficha ↔ necropsia confirmados y las tablas no contienen nombres que permitan construirlos. Las fichas ya están procesadas (`fichas_vector_search_gold`, 75 fichas con texto canónico y filtros) y las necropsias consolidadas (`necropsias_enriquecidas`). Las comparaciones se hacen directamente entre esas tablas reales.
- Consecuencias:
  - Salen la regresión logística y LambdaMART (requieren etiquetas). Entra TF-IDF + coseno. Quedan siete métodos no supervisados: reglas SQL, cadenas aproximadas, TF-IDF, BM25, Fellegi-Sunter con EM, embeddings con Vector Search y cross-encoder preentrenado sin ajuste.
  - Recall@k y MRR no se pueden calcular sin verdad conocida. La métrica principal sin etiquetas queda por definir con el autor.
  - El "ajuste fino" se hace sobre los hiperparámetros de cada método (pesos por faceta, parámetros de BM25, umbrales de Splink, modelo de embeddings).
- Alternativa descartada y motivo: Generar pseudo-fichas como en el Avance 3, o volver a extraer las necropsias con LLM. El autor las descartó para trabajar solo con los datos reales existentes.
- Participación de la IA: Decisión del autor. Claude Code había propuesto pseudo-fichas y extracción de necropsias (P-A y P-D). El autor las rechazó y pidió dejar la restricción por escrito para cualquier agente. Claude Code la registró en `CLAUDE.md`, en el skill del proyecto, en el plan y en esta bitácora.

Decisión 25: Usar la cobertura como métrica principal del Avance 4, medida sin etiquetas y comparable entre métodos
- Justificación: Sin verdad conocida y sin un perito disponible para validar en este momento, la cobertura es la métrica que se puede calcular para todos los métodos. Se define así: % de necropsias en las que el mejor candidato destaca claramente frente a los demás candidatos de esa misma necropsia.
  - Como cada método puntúa en una escala distinta (BM25 sin tope, coseno de 0 a 1, reglas que cuentan coincidencias), no se usa un umbral fijo de score. Se usa un z robusto dentro de cada consulta: (score del 1.º − mediana del resto) / (1.4826 × MAD del resto).
  - Umbral calibrado por simulación. Con z ≥ 3, un método de puro ruido "cubría" 16–19 % de las consultas. Con z ≥ 5 y al menos 15 candidatos por consulta, ≤ 2.5 % (0.1 % con 70 candidatos).
  - Secundarias: margen mediano entre el 1.º y el 2.º, cobertura con z ≥ 2, concordancia top-10 entre métodos y tiempos de entrenamiento e inferencia.
- Limitación declarada: la cobertura mide que el método encuentra una señal fuerte, no que el candidato sea correcto. La confirmación es pericial.
- Alternativa descartada y motivo: Precisión@5 según revisión pericial, porque no se cuenta con el perito en esta etapa. Concordancia entre métodos como métrica principal, porque es circular (premia parecerse a la mayoría).
- Participación de la IA: El autor eligió la cobertura. Claude Code propuso la definición con z robusto. La primera versión (z ≥ 3) falló la autoprueba del arnés: con scores al azar daba 21 % de cobertura. A raíz de eso, Claude Code simuló la distribución nula y ajustó el umbral a z ≥ 5 con mínimo 15 candidatos. El arnés actualizado pasó la autoprueba en Databricks.

Decisión 26: Elegir BM25 ajustado como modelo individual final del Avance 4
- Justificación: En la comparativa de los 7 modelos no supervisados sobre 884 necropsias reales contra 60 fichas (galería por facetas, blocking por sexo), los léxicos fueron los mejores:
  - Cobertura: TF-IDF 0.602, BM25 0.469, Splink 0.356, cross-encoder 0.317, reglas SQL 0.205, fuzzy 0.175, embeddings 0.000 y azar 0.000.
  - Ajuste fino con Optuna, 50 intentos por finalista, optimizando en una mitad de las necropsias y validando en la otra: BM25 llegó a 0.699 en validación (IC 95 % 0.655–0.745) y TF-IDF a 0.609 (0.566–0.657).
  - Los intervalos apenas se tocan. El desempate por interpretabilidad (contribución por palabra coincidente, visible para el perito), por costo (0.5 ms por consulta) y por la mejora en validación igual a la de ajuste coincide con BM25.
- Limitaciones declaradas:
  - La cobertura mide señal, no acierto.
  - El ajuste bajó a ≈ 0.12 el peso de los tatuajes porque casi no se registran en las necropsias. La salida al perito debe mostrar siempre las coincidencias de tatuajes.
  - k1 y los pesos de marcas quedaron en el borde de su rango de búsqueda.
- Alternativa descartada y motivo: TF-IDF, con menor cobertura en validación y menos legible para el perito. Los embeddings densos y el cross-encoder, sin señal suficiente o con un costo unas 20 mil veces mayor por consulta.
- Participación de la IA: Claude Code escribió el notebook único `Avance_4_equipo23` (arnés, 7 modelos, ajuste fino y elección), lo ejecutó como job en Databricks y redactó la justificación con los resultados. Hubo tres incidencias que la IA corrigió:
  - La versión de MLflow del cómputo serverless no aceptaba `parent_run_id`.
  - El endpoint de embeddings limitó las peticiones (HTTP 429); se agregaron reintentos.
  - El texto de `fichas_vector_search_gold` no describe prendas: Claude Code lo detectó en el código del notebook 03, lo verificó con conteos (0 de 75 fichas mencionan prendas) y lo comprobó empíricamente (reglas SQL con cobertura 0 en esa galería). Por eso se usó la galería por facetas.

  El autor eligió la métrica, aprobó los modelos y pidió que todo quedara en un solo notebook.

Decisión 27: Entregar como Avance 4 el notebook de Jaqueline (`Notebooks/Avance_4_Modelos_Alternativos_equipo23.ipynb`), con sus resultados originales, y conservar el trabajo no supervisado sobre datos reales como línea de validación hacia la Fiscalía
- Justificación: Con la rúbrica del avance (Comparativa 60, Ajuste fino 30, Modelo individual 10), el notebook de Jaqueline obtiene la mayor calificación estimada (≈ 92–100 frente a ≈ 83–93; ver la revisión siguiente). Usa una métrica convencional (Recall@10), ocho familias de clasificadores y un ajuste fino que incluye tasa de aprendizaje, activación y arquitectura de un MLP.
- Alcance de la decisión: aplica al entregable académico. Implica que el entregable usa un enfoque supervisado con pseudo-fichas, lo que contradice la Decisión 24 para este avance. Ambos enfoques se reportan con su alcance:
  - El supervisado con pseudo-fichas mide el techo del cotejo cuando la ficha se parece al expediente.
  - El no supervisado sobre datos reales mide la señal que existe hoy entre fichas y necropsias reales.
- Alternativa descartada y motivo: Entregar el notebook no supervisado (`…_forensic_match.ipynb`). Describe con más fidelidad los datos reales, pero su métrica principal (cobertura) mide señal y no acierto, y su ajuste fino cubre solo dos modelos sin red neuronal, lo que lo pone por debajo en la rúbrica.
- Condición pendiente: el notebook de Jaqueline se entrega sin modificar sus resultados. Sus salidas incluyen tres tablas con cuasi-identificadores de expedientes reales (sexo, edad, estatura, municipio). Se recomienda que la autora las regenere enmascaradas antes de compartir el archivo fuera del equipo.
- Participación de la IA: Claude Code estimó la calificación de ambos notebooks y recomendó el de Jaqueline. La decisión la tomó el autor.

### Revisión de los entregables del Avance 4 contra la rúbrica (hecha por la IA)

El autor pidió a Claude Code comparar los dos notebooks del Avance 4 contra la rúbrica (Comparativa 60, Ajuste fino 30, Modelo individual 10). La revisión leyó la estructura, el código y los tipos de salida. No abrió filas ni textos de las salidas. Es una estimación de la IA, no una calificación oficial. La elección del entregable corresponde al equipo.

| Criterio | `Avance_4_Modelos_Alternativos_equipo23.ipynb` (Jaqueline) | `…_forensic_match.ipynb` (Edmundo), primera versión sin salidas | `…_forensic_match.ipynb` (Edmundo), versión con salidas (25 oct, 08:40) |
|---|---|---|---|
| Comparativa (60) | 56–60 | 30–40 | 52–57 |
| Ajuste fino (30) | 27–30 | 12–18 | 23–27 |
| Modelo individual (10) | 9–10 | 2–4 | 8–9 |
| Total estimado | ≈ 92–100 | ≈ 45–60 | ≈ 83–93 |

- Notebook de Jaqueline: 8 clasificadores individuales sobre un vector de comparación de 10 características, evaluados con pseudo-fichas. Tiene tabla visible ordenada por Recall@10, seis métricas, tiempos e IC. Ajusta 3 modelos con 300 ajustes cada uno, incluido un MLP con tasa de aprendizaje, activación y arquitectura. Elige el modelo final con una matriz de decisión ponderada (regresión logística, Recall@10 = 0.869).
  - Observaciones: es un enfoque supervisado con pseudo-fichas, que contradice la Decisión 24. Su Recall@10 es un techo optimista y no el desempeño con fichas reales.
  - Privacidad: tres tablas de salida con cuasi-identificadores de expedientes reales (sexo, edad, estatura, municipio) que deben limpiarse antes de entregar.
- Notebook de Edmundo: 7 modelos no supervisados sobre datos reales, con cobertura como métrica principal y Optuna sobre los 2 mejores (modelo final BM25, Decisión 26).
  - La primera versión en el repositorio era el clon del job sin salidas (Databricks no las incluye al hacer commit desde una carpeta Git) y con la sección 8 provisional. El autor la reemplazó por una versión con salidas y justificación completa, y la IA la recalificó.
  - Las salidas solo contienen agregados: conteos, tabla comparativa, tabla de ajuste y gráficas. No hay filas reales.
  - Pendientes que restan puntos: las curvas de convergencia y la importancia de hiperparámetros quedaron solo en MLflow; hay ruido de instalación y advertencias de Splink en las salidas; la métrica es poco convencional (mide señal, no acierto); solo se ajustan dos modelos, sin red neuronal.
- Conclusión de la revisión: el notebook de Jaqueline obtendría mejor calificación con esta rúbrica. El de Edmundo describe con más fidelidad lo que los datos reales permiten hoy.
- Participación de la IA: Claude Code realizó la revisión y la estimación de puntajes. También es autor de buena parte del código del notebook de Edmundo, lo que se declara como posible sesgo de la revisión.

### Propuestas pendientes de aprobación (Avance 4)

Las redactó Claude Code en el plan `docs/plan_avance4.md`. Cuando el equipo las apruebe, rechace o modifique, pasan a ser decisiones numeradas.

- ~~P-A~~ (descartada por el autor, Decisión 24): Extraer las necropsias con el **mismo esquema Pydantic** que las fichas (LLM de solo texto sobre ropa, pertenencias y tatuajes) antes de comparar. Motivo: hoy se compara texto canónico contra texto libre, lo que explica los scores planos (0.52–0.59) del notebook `05_matching`
- P-B: Comparar **por faceta**, con emparejamiento elemento por elemento (prenda contra prenda, tatuaje contra tatuaje), en lugar de un solo embedding del texto completo.
- P-C: Tratar los pesos por faceta (0.65/0.25/0.10 del notebook 06) como **hiperparámetros o pesos aprendidos**, no como valores fijos a mano.
- ~~P-D~~ (descartada por el autor, Decisión 24): Generar pseudo-fichas desde los 251 identificados no reclamados que tienen ropa descrita, con ruido en tres niveles y partición agrupada por necropsia. Galería = 60 fichas reales + pseudo-fichas.
- P-E: Blocking: mismo sexo, `fecha_hechos ≤ fecha_ingreso` y fetos excluidos de la consulta. La edad se usa solo en los 58 registros que la tienen.
- P-F: Hacer de la **vestimenta la faceta principal**, con pertenencias como faceta propia y tatuajes como señal secundaria. Reportar la escasez de tatuajes como recomendación de captura para el SEMEFO.

### Trabajo en Databricks pendiente de documentar (participación de la IA por confirmar)

- Extracción v2: todas las páginas de un PDF en una sola llamada al LLM, porque procesar página por página generaba fichas alucinadas.
- Elección del modelo de embeddings `databricks-qwen3-embedding-0-6b` y del endpoint de Vector Search `forensic`.
- Índices separados por faceta (`vs_tatuajes_idx`, `vs_vestimenta_idx`, `vs_senas_idx`) y los pesos 0.65/0.25/0.10.
- Normalización de sexo en necropsias, que prefiere la tabla de 2026 cuando reporta HOMBRE o MUJER.
- El endpoint `gema` sirve Claude Sonnet 4.5 (según `resumen_proyecto.md`); confirmar y agregarlo en la tabla de herramientas.

## 8. Resumen del Avance 4: resultado entregado y camino recorrido

### 8.1 Resultado entregado (notebook de Jaqueline)

`Notebooks/Avance_4_Modelos_Alternativos_equipo23.ipynb`, autora Jaqueline Villafranca Hernández. Cotejo ficha → expediente evaluado como ranking sobre 107 pseudo-fichas de prueba (64,008 pares), con blocking (techo del protocolo: Recall@10 = 0.981) y un vector de comparación de 10 características.

**Comparativa de los ocho modelos individuales** (ordenada por Recall@10, métrica principal):

| Modelo | Recall@10 | Recall@1 | MRR | PR-AUC | Entrenamiento (s) | Inferencia por ficha (ms) |
|---|---|---|---|---|---|---|
| Regresión logística | **0.869** | 0.766 | 0.802 | 0.581 | 0.04 | 0.02 |
| SVM lineal | 0.860 | 0.738 | 0.781 | 0.483 | 0.08 | 0.01 |
| Perceptrón multicapa (MLP) | 0.860 | 0.729 | 0.784 | 0.645 | 4.28 | 0.06 |
| k-NN | 0.841 | 0.682 | 0.742 | 0.662 | 0.04 | 12.24 |
| Naive Bayes gaussiano | 0.841 | 0.645 | 0.710 | 0.381 | 0.01 | 0.03 |
| SVM RBF (Nyström) | 0.841 | 0.570 | 0.664 | 0.254 | 2.17 | 0.38 |
| LDA | 0.832 | 0.710 | 0.752 | 0.304 | 0.04 | 0.01 |
| Árbol de decisión (CART) | 0.766 | 0.589 | 0.647 | 0.501 | 0.10 | 0.01 |
| *Piso: heurística TF-IDF* | *0.850* | *0.766* | *0.801* | *0.521* | — | — |
| *Piso: azar* | *0.028* | *0.000* | *0.014* | *0.002* | — | — |

**Ajuste fino** (60 configuraciones × 5 pliegues de `GroupKFold` = 300 ajustes por modelo):

| Modelo | Recall@10 antes | Recall@10 después | IC 95 % |
|---|---|---|---|
| Regresión logística | 0.869 | 0.869 (se conserva la configuración preliminar) | 0.804–0.925 |
| SVM lineal | 0.860 | 0.850 | 0.785–0.916 |
| MLP | 0.860 | 0.841 | 0.775–0.907 |

**Robustez al ruido de la ficha** (Recall@10, ruido bajo / medio / alto): regresión logística 0.925 / 0.850 / 0.710; SVM lineal 0.907 / 0.804 / 0.701; MLP 0.888 / 0.794 / 0.692; heurística TF-IDF 0.916 / 0.794 / 0.692.

**Modelo final: regresión logística.** Los IC de los finalistas se traslapan. Desempató una matriz ponderada: Recall@10 en el límite inferior del IC 30 %, interpretabilidad 25 %, Recall@1/MRR 15 %, robustez 15 %, costo 10 %, calibración 5 %. Puntajes: regresión logística 4.98, SVM lineal 3.00, MLP 1.40. Se cumple el umbral de negocio (Recall@10 ≥ 0.60), pero no la meta de 0.97. Conclusión de la autora: el límite está en la representación (10 características), no en el algoritmo.

### 8.2 Camino recorrido para llegar al entregable (con participación de la IA)

| Etapa | Qué se hizo | Quién decidió / ejecutó | Resultado |
|---|---|---|---|
| Revisión del repositorio | Inventario de notebooks, duplicados y datos sensibles en el historial | Claude Code revisó; el autor pidió empezar de nuevo | Plan de reorganización y alerta de privacidad |
| Fuente post-mortem | Consolidación `necropsias_enriquecidas` (1,246 registros) | Equipo (notebook 05a); IA revisó | Decisión 19 |
| Dirección de la consulta | Necropsia → fichas | Autor | Decisión 20 |
| Perfilado P0 | Conteos agregados en Databricks: ropa en 68 % de necropsias, tatuajes en 12, edad en 58 | IA diseñó y ejecutó consultas de solo agregados | Decisión 21; la vestimenta pasa a ser la faceta principal |
| Arnés y experimento | Notebook común de evaluación y experimento MLflow `/Shared/proyecto_mna/avance4_cotejo` | IA escribió y probó con datos sintéticos; autor aprobó | Decisiones 22 y 23 |
| Restricción no supervisada | Sin pares confirmados: sin modelos supervisados ni pseudo-fichas en el enfoque de datos reales | Autor | Decisión 24 |
| Métrica sin etiquetas | Cobertura con z robusto ≥ 5, calibrada por simulación (con z ≥ 3 el azar daba 21 %) | Autor eligió la métrica; IA la definió y calibró | Decisión 25 |
| Galería de fichas | La tabla gold no describe prendas (0 de 75); se usó la galería por facetas (60 fichas) | IA lo detectó y verificó; reglas SQL con cobertura 0 en la gold | Registrado en la Decisión 26 |
| Siete modelos no supervisados | SQL, fuzzy, TF-IDF, BM25, Splink (EM), embeddings qwen3 y cross-encoder sobre 884 necropsias reales | IA escribió y ejecutó como job; corrigió incompatibilidad de MLflow y límite 429 | Cobertura: TF-IDF 0.602, BM25 0.469, Splink 0.356, cross-encoder 0.317, SQL 0.205, fuzzy 0.175, embeddings 0.000, azar 0.000 |
| Ajuste fino no supervisado | Optuna, 50 intentos × 2 finalistas, ajuste en mitad A y validación en mitad B | IA escribió y ejecutó | BM25 ajustado: cobertura en validación 0.699 (IC 0.655–0.745); Decisión 26 |
| Comparación de entregables | Estimación de calificación de ambos notebooks contra la rúbrica | IA estimó; autor eligió | Decisión 27: se entrega el notebook de Jaqueline |
| Privacidad del repositorio | Purga del historial de git (fotos de fichas y libro de necropsias) | Autor autorizó; IA reescribió el historial local | Pendiente el force push a GitHub y la solicitud a GitHub Support |

### 8.3 Qué aporta cada enfoque y qué sigue

- **Notebook entregado (supervisado, pseudo-fichas):** responde qué tan bien se encontraría el expediente si la ficha se pareciera a él. Recall@10 = 0.869 es un techo optimista: las pseudo-fichas se generan a partir del propio expediente.
- **Trabajo no supervisado sobre datos reales (`…_forensic_match.ipynb` y `Notebooks/Databricks/Avance_4_equipo23.py`):** responde si hoy existe señal entre fichas y necropsias reales. BM25 propone un candidato que destaca en ~70 % de las necropsias evaluables y deja 620 necropsias no identificadas con su top-10 para revisión pericial (tabla `candidatos_top10_modelo_final` en Unity Catalog).
- **Hallazgos comunes:** los métodos léxicos y lineales igualan o superan a los más complejos (MLP, cross-encoder, embeddings). Ambos notebooks apuntan a que el límite está en la representación y en la captura de datos, sobre todo la escasez de tatuajes registrados en el SEMEFO.
- **Siguiente paso, acordado con el autor:** validar contra la realidad con peritos. Para eso hacen falta pares ficha ↔ necropsia confirmados (resultados de búsquedas concluidas) y datos más ricos de la operación forense. Con ellos se calcula Recall@k sobre casos reales y se calibra la confianza que se muestra al perito.
