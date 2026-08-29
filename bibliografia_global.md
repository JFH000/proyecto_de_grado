# Bibliografía global — glosario de términos y abreviaciones

Glosario transversal del proyecto: cada término se registra como **abreviatura → significado (forma completa) → definición**. Es un documento vivo — cada vez que se lea/resuma un nuevo paper con abreviaciones que no estén aquí, se agregan al final de la sección correspondiente (o se crea una nueva sección temática si no encaja en las existentes), en vez de duplicarlas en cada resumen individual de `knowledge/`.

Ejemplo de formato de entrada:

| Abreviatura | Significado | Definición |
|---|---|---|
| RL | Reinforcement Learning | Paradigma de aprendizaje en el que un agente aprende una política de acción por prueba y error, maximizando una señal de recompensa acumulada obtenida al interactuar con un entorno, en vez de aprender de ejemplos etiquetados. |

## Aprendizaje por refuerzo (RL) y variantes

| Abreviatura | Significado | Definición |
|---|---|---|
| RL | Reinforcement Learning | Paradigma de aprendizaje en el que un agente aprende una política de acción por prueba y error, maximizando una recompensa acumulada obtenida al interactuar con un entorno. |
| DRL | Deep Reinforcement Learning | RL en el que la política y/o la función de valor se aproximan con una red neuronal profunda en vez de con una tabla, lo que permite manejar espacios de estado/acción de alta dimensión. |
| MARL | Multi-Agent Reinforcement Learning | Extensión de RL a entornos con múltiples agentes que actúan simultáneamente; el estado siguiente y la recompensa de cada agente dependen de la acción conjunta de todos. |
| FRL | Federated Reinforcement Learning | RL (o MARL) en el que varios agentes entrenan modelos localmente con sus propios datos/experiencias y periódicamente agregan (federan) sus parámetros en vez de compartir datos crudos. |
| MDP | Markov Decision Process | Formalismo matemático (estado, acción, función de transición, recompensa) que modela el problema de decisión secuencial que RL busca resolver. |
| CTDE | Centralized Training with Decentralized Execution | Paradigma de MARL donde el entrenamiento usa información centralizada (p. ej. un crítico que ve el estado/acciones de todos los agentes), pero cada agente ejecuta su política usando solo su observación local. |
| DTE | Decentralized Training and Execution | Paradigma de MARL donde cada agente entrena y ejecuta su política usando únicamente su propia experiencia local, sin acceso a datos/observaciones de otros agentes durante el entrenamiento. |
| SGD | Stochastic Gradient Descent | Método de optimización que actualiza los parámetros de un modelo usando el gradiente estimado sobre un mini-lote de muestras en vez de sobre el conjunto de datos completo. |
| TD (error) | Temporal-Difference error | Diferencia entre el valor estimado de un estado/acción y una mejor estimación calculada un paso después (recompensa observada + valor futuro estimado); es la señal de error que actualiza las funciones de valor en Q-learning y sus variantes. |

## Deep Q-Learning y variantes usadas en FedMARL (Li et al., 2022)

| Abreviatura | Significado | Definición |
|---|---|---|
| DQN | Deep Q-Network | Red neuronal que aproxima la función de valor-acción Q(s,a) de Q-learning, permitiendo aplicar Q-learning a espacios de estado de alta dimensión. |
| DDQN | Double Deep Q-Network | Variante de DQN que usa una red separada (o desacoplada) para seleccionar la mejor acción y para evaluarla, reduciendo el sesgo de sobreestimación del valor Q que tiene el DQN estándar. |
| D3QN | Dueling Double Deep Q-Network | DDQN que además separa la salida de la red en dos subredes — valor de estado V(s) y ventaja de acción A(s,a) — combinadas al final para estimar Q(s,a); es la arquitectura que cada agente V2V entrena localmente en Li et al. (2022). |
| PER | Prioritized Experience Replay | Técnica de repetición de experiencia en la que las transiciones (s,a,r,s') con mayor error TD se muestrean con más frecuencia del buffer de repetición, en vez de muestrear uniformemente. |
| ε-greedy | Epsilon-greedy | Estrategia de exploración-explotación usada por la red "evaluate" del D3QN: con probabilidad 1−ε el agente toma la acción que maximiza el valor Q estimado, arg maxₐ Q(s,a;θₑ); con probabilidad ε toma una acción aleatoria en su lugar, para seguir explorando el espacio de acciones en vez de explotar siempre la política actual. |
| 3M+1 | — | Dimensión del vector de estado que recibe cada agente V2V (M = número de canales/CUEs): M componentes del coeficiente de canal de su propio enlace V2V, M del coeficiente de canal de cada CUE, M de la selección de canal hecha por sus vecinos en el slot anterior, y +1 por la longitud de su propia cola de transmisión Q_k^t (Sec. IV-A, "Agent Design"). |

## Federated Learning (FL)

| Abreviatura | Significado | Definición |
|---|---|---|
| FL | Federated Learning | Esquema de entrenamiento distribuido en el que varios clientes entrenan un modelo localmente sobre sus propios datos y solo comparten los parámetros/actualizaciones del modelo con un agregador (central o descentralizado), sin transmitir los datos crudos. |
| FedAvg | Federated Averaging | Algoritmo de FL en el que el agregador combina los modelos locales calculando un promedio (ponderado, p. ej. por tamaño de mini-lote o de dataset) de sus parámetros, y redistribuye el modelo global resultante a los clientes. |
| FedMARL | Federated Multi-Agent Reinforcement Learning | Nombre que da Li et al. (2022) a su esquema, que combina agentes MARL entrenados de forma DTE con una capa de FedAvg para sincronizar periódicamente sus modelos. |

## Comunicaciones vehiculares (V2V / V2X) — contexto del paper de Li et al. (2022)

| Abreviatura | Significado | Definición |
|---|---|---|
| V2V | Vehicle-to-Vehicle | Modo de comunicación directa entre vehículos, usado típicamente para intercambiar información de seguridad vial de baja latencia. |
| V2X | Vehicle-to-Everything | Término paraguas para toda comunicación vehicular (con otros vehículos, con infraestructura, con la red celular, con peatones, etc.), del cual V2V es un caso particular. |
| C-V2X | Cellular V2X | Estándar de V2X que se apoya en la infraestructura de red celular (en vez de en DSRC), permitiendo mayor cobertura y capacidad de comunicación. |
| DSRC | Dedicated Short-Range Communications | Uno de los dos estándares principales para comunicación V2X (junto con C-V2X), basado en comunicación directa de corto alcance sin apoyarse en la infraestructura de red celular. El paper señala que C-V2X lo supera en cobertura, capacidad de comunicación y resiliencia a la interferencia (Sec. I). |
| D2D | Device-to-Device | Interfaz de comunicación directa entre dispositivos sin pasar por la estación base, usada en C-V2X para permitir los enlaces V2V. |
| CUE | Cellular User (Equipment) | Usuario celular al que ya se le asignó un canal ortogonal fijo en la celda; los pares V2V deben reutilizar estos canales sin degradar su servicio. |
| VUE | Vehicular User Equipment | Equipo/terminal del usuario vehicular (transmisor o receptor de un enlace V2V). |
| BS | Base Station | Estación base celular; en Li et al. (2022) actúa además como agregador central del esquema FedAvg. |
| SINR | Signal-to-Interference-plus-Noise Ratio | Relación entre la potencia de la señal deseada y la suma de la interferencia más el ruido; se usa para medir la confiabilidad de un enlace y su probabilidad de outage. |
| QoS | Quality of Service | Conjunto de requisitos de desempeño (tasa mínima, retardo máximo, confiabilidad, etc.) que un enlace de comunicación debe cumplir. |
| AWGN | Additive White Gaussian Noise | Modelo de ruido térmico del canal, de potencia constante y distribución gaussiana, sumado a la señal recibida. |
| LOS / WLOS / NLOS | Line-of-Sight / Weak-LOS / Non-Line-of-Sight | Clasificación del tipo de trayecto de propagación entre transmisor y receptor según haya o no obstrucción directa, usada para modelar el desvanecimiento a gran escala del canal. |
| OFDM | Orthogonal Frequency-Division Multiplexing | Esquema de modulación que divide el ancho de banda en subportadoras ortogonales (canales), permitiendo su asignación independiente a distintos usuarios. |
| 3GPP | 3rd Generation Partnership Project | Organismo de estandarización de redes celulares; sus reportes técnicos (p. ej. TR 36.885) definen parámetros como el radio de vecindad V2V o el retardo máximo tolerable usados en el modelo del paper. |

## Baselines/paradigmas de FL citados por comparación

| Abreviatura | Significado | Definición |
|---|---|---|
| DFL | Decentralized Federated Learning | Variante de FL sin agregador central: los clientes intercambian y promedian modelos directamente entre pares (p2p/gossip), en contraste con el esquema cliente-servidor de FedAvg usado en Li et al. (2022). |
| HFRL | Horizontal Federated Reinforcement Learning | Variante de FRL en la que cada agente posee su propia copia independiente del entorno/dataset con las mismas variables (features), y se federa para compartir conocimiento entre entornos similares. |
| VFRL | Vertical Federated Reinforcement Learning | Variante de FRL en la que los agentes comparten un mismo entorno pero cada uno observa una parte distinta del espacio de estado/features, fusionando representaciones intermedias (típicamente de forma cifrada) en vez de promediar pesos completos. |
| SADRL | Single-Agent DRL | Baseline experimental de Li et al. (2022) en el que solo un par V2V realmente entrena su modelo DRL y el resto simplemente replica ese modelo — equivalente a FedAvg con K=1 agente real aprendiendo. |
| GM | Greedy Method | Baseline experimental de Li et al. (2022): cada V2V elige siempre el canal de menor interferencia y la máxima potencia disponible, sin aprendizaje. |
| CM | Clustering Method | Baseline experimental de Li et al. (2022): los V2V se agrupan en clústeres geográficos fijos y la asignación de canal se resuelve dentro de cada clúster mediante un algoritmo de matching (deferred-acceptance). |

## Paradigmas de entrenamiento multiagente (complemento a CTDE/DTE)

| Abreviatura | Significado | Definición |
|---|---|---|
| CTE | Centralized Training and Execution | Un único controlador centralizado actúa sobre el historial conjunto de todos los agentes tanto al entrenar como al ejecutar; equivale a resolver un solo MDP/POMDP sobre el espacio conjunto. No escala (espacio de acciones exponencial) pero es la cota superior de desempeño alcanzable. |
| CTCE / DTDE | (sinónimos de CTE / DTE) | Nombres alternativos usados por algunos surveys (p. ej. Yadav, Mishra & Kim, 2023) para exactamente los mismos conceptos de CTE (Centralized Training and *Centralized* Execution) y DTE (Decentralized Training and Decentralized Execution). |
| POMDP | Partially Observable MDP | Extensión del MDP donde el agente no observa el estado real, solo una observación parcial derivada de él. |
| Dec-POMDP | Decentralized POMDP | Versión multiagente del POMDP: cada agente tiene su propio historial de observaciones-acciones, y todos comparten una única recompensa conjunta (caso cooperativo). Formaliza el marco que usa Amato (2024) para definir CTE/DTE/CTDE. |
| No-estacionariedad | Non-stationarity | En MARL, el hecho de que varios agentes aprenden simultáneamente hace que el entorno "percibido" por cada uno cambie con el tiempo (porque los demás también cambian su política), rompiendo las garantías de convergencia del RL de un solo agente. |
| Lazy agent problem | — | Con una recompensa 100% compartida entre agentes, uno de ellos puede tener poco incentivo para aprender una buena política individual, ya que su bajo esfuerzo queda enmascarado por el desempeño de sus compañeros. Es una de las razones citadas para preferir CTDE sobre CTE/CTCE puro. |
| IGM | Individual-Global-Max | Propiedad que exige que el argmax del Q-valor conjunto coincida con la tupla de argmax de los Q-valores individuales de cada agente; es la condición formal que garantiza que factorizar el valor conjunto (VDN, QMIX) siga siendo consistente con actuar de forma descentralizada. |
| VDN / QMIX | Value Decomposition Networks / — | Familia de algoritmos CTDE que factorizan un Q-valor conjunto Qtot como función (aditiva en VDN, monótona no-lineal en QMIX) de los Q-valores individuales Qᵢ de cada agente, entrenados de punta a punta con una sola señal de TD conjunta. |
| MADDPG / COMA / MAPPO | — | Algoritmos CTDE de tipo actor-crítico: un crítico centralizado (ve historiales/acciones conjuntas) se usa solo para calcular el gradiente de política de cada actor descentralizado (que solo ve su propio historial); el crítico se descarta en ejecución. |

## Federated RL (FRL): variantes horizontal/vertical y descentralización

| Abreviatura | Significado | Definición |
|---|---|---|
| HFRL | Horizontal Federated RL | (Ver también entrada previa en "Baselines/paradigmas de FL".) Formalmente 𝒮ᵢ=𝒮ⱼ, 𝒜ᵢ=𝒜ⱼ, ℰᵢ≠ℰⱼ (Qi et al., 2021): agentes con los mismos espacios de estado/acción pero cada uno en su **propia copia independiente** del entorno; federan pesos/gradientes (típicamente vía FedAvg) sin compartir datos crudos. |
| VFRL | Vertical Federated RL | Formalmente 𝒪ᵢ≠𝒪ⱼ, 𝒜ᵢ≠𝒜ⱼ, ℰᵢ=ℰⱼ=ℰ (Qi et al., 2021): agentes que comparten un **único entorno**, cada uno con una vista/observación parcial distinta; en vez de compartir pesos, fusionan información parcial cifrada (embeddings, logits) en un valor/crítico compartido. Es el régimen federado con el mapeo más directo a CTDE, porque ambos asumen exactamente un entorno común con observabilidad parcial por agente. |
| CFL | Centralized Federated Learning | FL "clásico": un servidor central recibe los modelos/gradientes locales de todos los clientes, los agrega (p. ej. FedAvg) y redistribuye el modelo global. Punto único de falla y cuello de botella de comunicación. |
| DFL | Decentralized Federated Learning | (Ver también entrada previa.) Sin servidor: cada nodo agrega localmente los modelos recibidos de sus vecinos en la topología de red, y el conocimiento se propaga transitivamente por el grafo hasta converger. |
| SDFL | Semi-Decentralized FL | Punto intermedio entre CFL y DFL: el rol de agregador existe pero **rota** entre los participantes en vez de ser un nodo fijo o no existir. |
| Gossip | — | Mecanismo de comunicación descentralizado y típicamente asíncrono en el que cada nodo intercambia/promedia parámetros solo con un subconjunto de vecinos (fijos o re-muestreados cada ronda), sin ningún coordinador central ni necesidad de que todos los nodos participen en cada ronda. |

## Teoría de juegos aplicada a MA-FRL (Xu et al., 2022)

| Abreviatura | Significado | Definición |
|---|---|---|
| Juego de Stackelberg | — | Juego secuencial líder-seguidor: el líder mueve primero (en Xu et al., los edge nodes fijan un precio por mejora de modelo) y los seguidores responden maximizando su propia utilidad (los end devices eligen cuántas iteraciones locales entrenar); se resuelve por inducción hacia atrás y admite solución óptima única en cada etapa. |

---

*Fuente de esta primera tanda de términos: `knowledge/summary_fedmarl_v2v_resource_allocation.md` y la lectura directa de `papers/federated_marl/li_2022.pdf` (Li et al., 2022, "Federated Multi-Agent Deep Reinforcement Learning for Resource Allocation of Vehicle-to-Vehicle Communications"). Al procesar los demás papers de `papers/` y `knowledge/`, añadir aquí las abreviaciones nuevas que no estén ya cubiertas.*

*Añadido: **ε-greedy** — definido en `papers/federated_marl/li_2022.pdf` (Sec. IV-B, "Local Training of V2V Agents").*

*Añadido: **DSRC** — definido en `papers/federated_marl/li_2022.pdf` (Sec. I, Introducción).*

*Añadido: **3M+1** — definido en `papers/federated_marl/li_2022.pdf` (Sec. IV-A, "Agent Design"); confirmado también en `knowledge/summary_fedmarl_v2v_resource_allocation.md`.*

*Segunda tanda (preparación reunión 3): **CTE, CTCE/DTDE, POMDP, Dec-POMDP, no-estacionariedad, lazy agent problem, IGM, VDN, QMIX, MADDPG, COMA, MAPPO, HFRL, VFRL, CFL, DFL, SDFL, gossip, juego de Stackelberg** — definidos a partir de `knowledge/summary_cooperative_marl_intro.md` (Amato, 2024), `knowledge/summary_frl_techniques_survey.md` (Qi et al., 2021), `knowledge/summary_decentralized_fl_survey.md` (Martínez Beltrán et al., 2023), `knowledge/summary_marl_cav_survey.md` (Yadav, Mishra & Kim, 2023), `knowledge/summary_frl_secure_incentive_icps.md` (Xu et al., 2022), y las páginas de Notion "Transferibilidad CTDE ↔ FRL" y "Escenarios: drones y federado" (Ekechi et al., 2025).*
