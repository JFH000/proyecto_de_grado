# Resumen — Reunión 4 (Asesoría Proyecto de Grado)

**Fuente:** `reuniones/reu_4.txt`
**Participantes:** Nicolás (asesor), Estudiante

> Nota sobre el transcript: las etiquetas de hablante están invertidas. "S1 (Estudiante)" es en realidad el asesor (Nicolás) y "S2 (Profesor)" es el estudiante (S2 cierra con "Gracias Nicolás"). Además, "V2B" es un error de transcripción de **V2V**, y el transcript arranca a mitad de la conversación, ya comentando los resultados del primer experimento.

## Propósito de la reunión

Reunión de revisión del primer experimento (la réplica de Federated Learning horizontal sobre comunicaciones vehiculares acordada en la Reunión 3) y de definición del siguiente paso: pasar de replicar a proponer una **extensión** que pueda convertirse en el aporte de la tesis.

## Resultados del primer experimento

- La réplica de **Li et al. (2022)** funciona: el algoritmo federado entrena correctamente sobre **5 agentes**, donde cada agente es un **par V2V** (un vehículo transmisor y uno receptor).
- El paper escala de **5 a 30 agentes** para medir cómo se degradan la recompensa y la señal. El estudiante solo probó con 5, pero en ese punto los números quedan en un rango parecido al del paper.
- El asesor lo toma como señal de que, si con 5 agentes se está en el mismo rango, escalar a más agentes debería comportarse de forma similar. Es una suposición razonable, pero no se ha verificado.
- Solo se implementó la variante **horizontal** (no la vertical): cada agente explora localmente, los modelos se fusionan en un solo modelo en el punto central y ese modelo se redistribuye a todos los agentes.

## Punto débil identificado: la Base Station centralizada

En el paper, la **Base Station (BS)** concentra dos funciones:

1. **Agregación de modelos**: el promedio ponderado que produce el modelo global.
2. **Cálculo de la recompensa**: una recompensa global única para todos los agentes, que según el paper los incentiva a cooperar.

Desde esto el estudiante había planteado la idea que usó como base para la entrega de ética. El asesor coincide en que es "el punto más atacable" de la implementación.

## Opciones para descentralizar la agregación

El asesor propone tres alternativas, **en orden de complejidad creciente**:

| # | Estrategia | Idea | Consideraciones |
|---|---|---|---|
| 1 | **Promedio con vecinos** (propuesta del estudiante) | Tomada de un paper de clasificación de imágenes en IoT, sin FL centralizado: cada agente elige 2 o 3 vecinos al azar y promedia su modelo con los de ellos. En V2V, cada par ya se comunica con vecinos a ~300 m, así que la fusión ocurriría entre vecinos cercanos y el conocimiento se propagaría a medida que los vehículos se mueven. | Es la más accesible porque el estudiante ya leyó el paper base. Corresponde a los conceptos de **DFL** y **gossip** del glosario. |
| 2 | **Algoritmos de elección de líder** (sistemas distribuidos) | El agregador sigue existiendo, pero es uno de los propios vehículos. Si falla o sale del alcance, los demás eligen un nuevo líder que reconstruye el modelo conjunto. | Hay que aprenderla desde cero. |
| 3 | **CRDTs** (*Conflict-free Replicated Data Types*) | No hay modelo central: los agentes se comparten información para que **cada uno** construya el modelo general. No requiere sincronización, tolera fallos y un agente que se va y vuelve se pone al día con lo que aportaron los demás. | Cada agente debe poder guardar el modelo completo. En vehículos se asume que la memoria no es problema porque son "computadores muy grandes". Hay que aprenderla desde cero. |

## Decisión de enfoque

- El estudiante ve dos aportes posibles: (a) descentralizar **dónde se unen los modelos** y (b) descentralizar **la recompensa**.
- Sobre la recompensa, el asesor señala que si cada agente sigue de forma independiente **la misma función de recompensa**, como restricción de diseño, la teoría dice que el efecto debería ser equivalente a tener una política general conjunta.
- **No se van a atacar ambos.** Se elige **descentralizar el modelo (la agregación)**, que a los dos les parece lo más interesante.
- **Estrategia de trabajo**: empezar por la opción 1 (promedio con vecinos). Si los resultados o las mejoras no son buenos, probar otra de las estrategias.

## Alcance del aporte y rigor

- **Aporte**: una tesis debe tener un aporte, pero es muy difícil que sea significativo en 16 semanas. Desde el inicio el proyecto se planteó como exploratorio, así que **el aporte puede ser el estudio en sí**. Si al final del semestre la estrategia da malos resultados pero se entiende por qué y dónde falla, ese es un resultado válido. Si en cambio funciona rápido (mejor recompensa y menos comunicación que el modelo centralizado), se sigue extendiendo, porque queda semestre por delante.
- **Demostraciones formales**: no son necesarias. Basta con **validación experimental en simulación**, como en la mayoría de artículos y especialmente en una tesis. Si se puede demostrar algo, es un plus.
- **Metodología**: el estudiante comenta que no tiene claro cómo guiar el proyecto. El asesor lo tranquiliza: "científicamente estamos bien". Ya hay experimentos base comparables contra el estado del arte (el paper), y lo nuevo se comparará contra esos resultados base.

## Tareas asignadas

| Quién | Tarea |
|---|---|
| Estudiante | Explorar las opciones del asesor (líderes, CRDTs) y profundizar en su propia idea de promedio con vecinos. |
| Estudiante | Si alcanza el tiempo, ver cómo implementar la descentralización sobre el código ya existente (`v2v/`). |
| Estudiante | **Empezar a escribir el documento final**: descripción del algoritmo y resultados obtenidos hasta ahora, aprovechando los tiempos muertos (por ejemplo, mientras corren experimentos). |

## Logística / administrativo

- La entrega de **ética** ya fue subida.
- No hay más entregas en el sistema (SINFO) por ahora. Las siguientes son al **final del semestre (semanas 16–17)**.
- **Próxima reunión: en 15 días**, después del receso.

## Estado al cierre de la reunión

- Primer experimento cerrado con éxito: FedAvg horizontal replicado con 5 agentes, con resultados en un rango similar al de Li et al. (2022). Queda pendiente el escalamiento de 5 a 30 agentes.
- Dirección de la tesis definida: **eliminar el agregador central (BS)** y **descentralizar la agregación de modelos** entre vehículos, empezando por promedio con vecinos.
- Criterio de éxito acordado: comparar contra la línea base centralizada ya obtenida en recompensa y, si es posible, en costo de comunicación.

## Relevancia para el proyecto

Esta reunión cierra la fase de **réplica** iniciada en la Reunión 3 (ver [`resumen_reunion_3.md`](resumen_reunion_3.md)) y abre la fase de **extensión/aporte**. Lo que se replicó (DTE + FedAvg centralizado sobre pesos, ver `v2v/README.md`) pasa a ser la **línea base**. La pregunta de investigación se concreta en si el agregador central se puede reemplazar por un mecanismo descentralizado entre vehículos (gossip/DFL, elección de líder o CRDTs) sin perder desempeño, y con qué costo de comunicación. Esto retoma el eje de Federated Learning definido en la Reunión 2 (ver [`resumen_reunion_2.md`](resumen_reunion_2.md)), ahora desde la variante **descentralizada** (DFL) en lugar de la cliente-servidor.
