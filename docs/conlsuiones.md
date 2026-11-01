## Conclusiones (borrador — se completa al terminar los pasos 3 a 6)

**1. El cotejo es un sistema, no un modelo.** Encontrar el expediente de una persona desaparecida combina varias
técnicas en cadena: reglas determinísticas (blocking por sexo y edad, tipos de seña compartidos), similitud léxica
de textos (TF-IDF de n-gramas de caracteres sobre señas y ropa), un modelo que pondera esas pistas y, al final, el
perito que confirma. Ningún componente resuelve el problema por sí solo; el modelo sugiere y el perito decide.

**2. Las pistas más informativas son las señas particulares, la ropa y la edad.** La selección de características
(paso 1) coincide con el Avance 4: `sim_senas`, `sim_ropa` y `dif_edad` concentran la señal. El sexo actúa como
filtro previo (blocking), no como pista del modelo.

**3. Hasta ahora, el motor más simple es el más eficaz.** La regresión logística alcanza Recall@10 = 0.869 en prueba
(0.911 en validación cruzada). El Random Forest, aun con búsqueda de hiperparámetros, queda en 0.850 (0.903 en
validación cruzada): pierde de forma consistente en ambos exámenes, por lo que no es sobreajuste sino un motor menos
adecuado. Con solo 244 pares positivos de entrenamiento, la flexibilidad de los árboles no se aprovecha; la búsqueda
premió las configuraciones más simples (árboles de profundidad 6, hojas grandes) y el bosque resultó muy sensible a
sus hiperparámetros (0.766–0.903).
El Gradient Boosting confirma el patrón: Recall@10 = 0.841 en prueba (0.875 en validación cruzada), también por
debajo de la regresión logística en ambos exámenes. Su mejor configuración fue la más regularizada del espacio de
búsqueda (hojas de ≥ 200 pares, L2 = 10, árboles de 15 hojas) y su desempeño varió entre 0.617 y 0.875 según los
hiperparámetros. Su PR-AUC alta (0.676) indica que separa bien los pares correctos a nivel global, pero no ordena
mejor los candidatos dentro de cada ficha, que es lo que revisa el perito.

**4. Los modelos individuales tienen fortalezas complementarias.** La regresión logística coloca al correcto dentro
del top-10 con más frecuencia; el MLP lo coloca en primer lugar más seguido (Recall@1 0.701 vs. 0.673; PR-AUC 0.696
vs. 0.533). Esa complementariedad es la hipótesis que ponen a prueba los ensambles heterogéneos.

Ningún ensamble heterogéneo superó a la regresión logística en la métrica principal. La votación ponderada ganaba
una ficha en validación (0.919 vs. 0.911), pero en prueba quedó una por debajo (0.860 vs. 0.869): sus pesos se
ajustaron a casos particulares y la combinación ganadora no fue estable (varias combinaciones muy distintas
empataron). El stacking mejoró ligeramente el orden dentro de la lista (Recall@1 0.692 vs. 0.673; MRR 0.745 vs.
0.732), pero perdió una ficha en el top-10 y requiere 400 veces más tiempo de entrenamiento y mantener cuatro
modelos. Su meta-modelo eligió la máxima regularización. En las cinco búsquedas de hiperparámetros realizadas,
la configuración ganadora fue siempre la más simple.

**5. El límite está en la información, no en el algoritmo.** El blocking fija un techo de 0.953 (5 de 107 fichas
pierden su expediente antes de puntuar), y familias de modelos muy distintas convergen al mismo rango de desempeño.
Las mejoras grandes vendrán de mejores pistas (embeddings semánticos, extracción estructurada de tatuajes, captura
sistemática de señas en el SEMEFO), no de motores más complejos.

5.1 Elección del modelo final: regresión logística

Ninguno de los cuatro ensambles mejora a la regresión logística: todos los intervalos de confianza de la diferencia
pareada (bootstrap, B = 2,000) cruzan el cero, y en valor puntual todos quedan entre 1 y 3 fichas por debajo. Ante un
empate estadístico, la elección se decide por los criterios del proceso pericial:

| Criterio | Regresión logística | Mejor ensamble (stacking) |
|---|---|---|
| Recall@10 (métrica principal) | **0.869** [0.804, 0.925] | 0.860 [0.794, 0.925] |
| Recall@1 / MRR | 0.673 / 0.732 | **0.692 / 0.745** (+2 fichas en primer lugar, diferencia no significativa) |
| Tiempo de entrenamiento | **0.13 s** | 52.5 s (≈ 400 veces más) |
| Inferencia por ficha | **0.10 ms** | 0.76 ms |
| Explicabilidad ante el perito y el Ministerio Público | **Un peso por pista, legible como *odds ratio*** | Pesos sobre los scores de tres modelos; el MLP no es interpretable |
| Implementación en la base de datos del sistema real | **Una expresión SQL** (suma ponderada) | Requiere servir cuatro modelos fuera de la base de datos |
| Mantenimiento | Un modelo | Cuatro modelos y un esquema de predicciones fuera de pliegue |

**Argumentos de negocio:**
1. **El error más costoso es el falso negativo** (un expediente correcto fuera de la lista que revisa el perito). En
   esa métrica la logística es la mejor, aunque empatada con los ensambles.
2. **La sugerencia debe poder explicarse.** El perito firma un dictamen y el Ministerio Público puede preguntar por qué
   se revisó un expediente. La logística dice cuánto aportó cada pista; un ensamble no.
3. **Parsimonia.** Si la complejidad adicional no se traduce en una mejora demostrable, no se justifica su costo de
   entrenamiento, mantenimiento e infraestructura.
4. **Llevable al sistema real.** La logística se ejecuta como una consulta SQL junto a la búsqueda léxica y
   vectorial, sin infraestructura adicional.

**6. Modelo final y justificación.** 

La elección depende del caso de uso: para listas de candidatos por ficha la logística empata con los ensambles y
gana en simplicidad; para priorizar cruces entre todas las fichas (umbral global), el stacking separa mejor
(AP 0.681 vs. 0.533).

**7. Limitaciones.** Los resultados provienen de pseudo-fichas generadas con un ruido que nosotros definimos y de un
registro sintético calibrado con agregados del real; el Recall@10 es un techo optimista. Con 107 fichas de prueba,
diferencias de 1–2 fichas no son estadísticamente distinguibles.

**8. Implicaciones para el sistema real.** Cada componente tiene equivalente en una base de datos con capacidades
vectoriales (blocking con `WHERE`, similitud de trigramas con `pg_trgm`, embeddings con `pgvector`). Si el modelo
final es lineal, puede ejecutarse como una expresión SQL; sin pares confirmados, la fusión por rangos (RRF) permite
combinar búsquedas sin entrenar. Con cotejos confirmados por la Fiscalía, el mismo protocolo sustituye las
pseudo-fichas por casos reales y produce la cifra de desempeño verdadera.

