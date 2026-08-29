# Resumen — Reunión 3 (Asesoría Proyecto de Grado)

**Fuente:** `reuniones/reu_3.txt`
**Participantes:** Nicolás (asesor), Estudiante

## Propósito de la reunión

Reunión de definición de enfoque: tras la exploración bibliográfica de las reuniones anteriores, se decide por cuál camino concreto empezar a implementar, y se acuerda el primer experimento práctico de la tesis. El transcript disponible arranca a mitad de la conversación (ya en la comparación entre dos opciones), por lo que no se capturó el planteamiento inicial de esas opciones.

## Decisión de enfoque

- Se compararon dos caminos posibles para la tesis (el detalle de la "opción 1" no quedó registrado en el transcript, pero corresponde a trabajo relacionado con **seguridad/aseguramiento**, de interés para la continuidad del grupo de investigación más allá de la tesis):
  1. **Camino de seguridad**: más atractivo para el asesor a largo plazo ("egoístamente... esa es la más chévere"), porque conecta con otra línea de trabajo del grupo y podría dar pie a seguir trabajando después de la tesis.
  2. **Federated Learning (horizontal)**: más conveniente pensando en que el estudiante se gradúe — es más simple de construir en términos de código ("tiene menos cosas"), aunque el asesor aclara que "todo va a ser difícil".
- **Se opta por el camino 2**: empezar por **Federated Learning horizontal**, usando uno de los escenarios ya revisados (carros o drones).
- El asesor tiene preferencia personal por el ejemplo de **carros/vehículos**: menos dimensiones, más fácil de manejar. Aclara que es solo su preferencia — al ser la tesis del estudiante, puede elegir el escenario que prefiera.

## Primer experimento acordado

- **Tarea concreta**: replicar, a partir de un paper ya revisado, una implementación de Federated Learning horizontal sobre un escenario existente (vehículos o drones), con resultados claros y demostrables ("esto funciona así, y tenemos estos resultados").
- El estudiante describe el paper de vehículos que tiene en mente: no trata de vehículos conduciendo/interactuando en tráfico, sino de vehículos **comunicándose a través de una red** vía Reinforcement Learning — un vehículo transmisor aprende **con qué red comunicarse** y **con qué potencia/alcance transmitir** hacia un receptor.
- Nicolás valida esta idea como buen primer experimento: sirve para "meterse en la implementación" y verificar que las cosas funcionan, aunque nota que es "un caso más localizado" frente a otro problema detectado en la sesión (no se alcanza a detallar cuál).
- **Stack de implementación**: Python, priorizando **reutilizar librerías o bases de implementación existentes** en vez de construir todo desde cero.

## Logística / administrativo

- El estudiante compartirá el avance mediante un **repositorio** de código.
- El asesor estará fuera (conferencia) las próximas **dos semanas**, sin disponibilidad para reunirse ni a los 8 ni a los 15 días; comunicación mientras tanto por **correo** (papers, dudas, avances).
- **Próxima reunión: en tres semanas** (a partir de la fecha de esta reunión).
- El estudiante planea explorar el entrenamiento y prevé posible necesidad de **GPU**; tiene acceso a **Hypatia** (servidor de la universidad) por un proyecto de un curso anterior. Si no logra usarlo, el asesor ofrece ayudar a gestionar tiempo en el servidor de la universidad.

## Estado al cierre de la reunión

- Enfoque definido: **Federated Learning horizontal** como primer camino de tesis, con un escenario de vehículos/drones como punto de partida.
- Primer entregable claro: una implementación replicada (no necesariamente original) que demuestre resultados funcionando, apoyada en librerías existentes.
- Sin reunión de seguimiento hasta dentro de tres semanas; avances y dudas se manejan por correo y repositorio.

## Relevancia para el proyecto

Esta reunión marca el cierre de la fase puramente exploratoria de las Reuniones 1 y 2 (ver [`resumen_reunion_1.md`](resumen_reunion_1.md) y [`resumen_reunion_2.md`](resumen_reunion_2.md)) y el arranque de la fase de **implementación**: de los dos ejes de trabajo identificados antes (caracterización de escenarios multi-agente y transferibilidad de técnicas CTE/DTE a Federated Learning), se elige concretar primero el eje de Federated Learning horizontal, usando el escenario de comunicación vehicular como banco de pruebas inicial.
