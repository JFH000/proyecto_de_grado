# Resumen — Reunión 1 (Asesoría Proyecto de Grado)

**Fuente:** `reuniones/reu_1.txt`
**Participantes:** Speaker 1 (asesor, Nicolás), Speaker 2 (estudiante)

## Propósito de la reunión

Reunión inicial de arranque del proyecto de grado. El objetivo fue definir el enfoque exploratorio inicial y repartir tareas de lectura antes de empezar a delimitar el problema concreto.

## Idea central del proyecto

- El tema propuesto por el asesor gira en torno a los **Federated Multi-Agent (Reinforcement Learning) Systems**: sistemas de múltiples agentes que aprenden en **entornos distribuidos**, combinando ideas de *Multi-Agent Systems* con *Federated Learning*.
- El resultado de la tesis puede tomar **dos caminos posibles**, a definir según lo que revele la exploración de literatura:
  1. Concluir que ese tipo de sistema **no existe aún** en la literatura → oportunidad de **construirlo/proponerlo**.
  2. Concluir que **sí existe** → generar una **implementación propia** con un entendimiento claro y profundo de cómo funciona.
- No es necesario inventar algo nuevo desde cero: es válido tomar ideas de uno o varios papers existentes ("nos robamos ideas de un par de papers") y que la **contribución de la tesis sea la implementación** (una plataforma/sistema para entrenar agentes en entornos distribuidos, aplicado a un problema específico).

## Tareas asignadas

| Quién | Tarea |
|---|---|
| Asesor (Speaker 1) | Enviar un paper *survey* general de Multi-Agent RL (colaborativo vs. competitivo, técnicas por categoría) + referencias sobre aprendizaje descentralizado + referencias de Multi-Agent en sistemas federados. Enviarlas el mismo día a las 5pm. |
| Estudiante (Speaker 2) | Empezar a explorar por su cuenta literatura de Multi-Agent Systems (reconoce no tener mucho conocimiento previo del tema). Redactar un borrador de la propuesta formal (documento corto, "cuatro líneas reales") y enviarlo al asesor para revisión. |
| Ambos | Leer la literatura enviada/encontrada antes de la próxima reunión. |

## Puntos técnicos mencionados (a explorar)

- Relación entre el **número de agentes** y el **rendimiento/velocidad de aprendizaje**: a más agentes, el desempeño del entrenamiento puede degradarse (mencionado como fenómeno conocido desde los años 90).
- Necesidad de entender **qué características se requieren** para poder implementar un sistema de agentes federado, y **hasta qué punto es viable** dado el tamaño del problema.

## Logística / administrativo

- Debe entregarse una **propuesta formal** (deadline percibido por el estudiante: 31 de agosto, aunque el asesor aclara que hay tiempo pero conviene resolverla rápido, "tarea de 5 minutos").
- La propuesta se trata como un trámite inicial ligero, no como el documento definitivo — sirve para "salir del paso" mientras se afina el enfoque real.
- Se resolvió una duda administrativa sobre el bloque/asignación del profesor en el sistema (ya estaba corregido).
- **Próxima reunión: en 15 días** (a partir de la fecha de esta reunión), para revisar avances de lectura y empezar a **dar foco / trazar ruta** concreta del proyecto.

## Estado al cierre de la reunión

- No hay aún objetivo formal ni alcance definido — la tarea explícita es "leer literatura para poder plantear un objetivo".
- No hay referencias específicas compartidas todavía en la reunión misma (el asesor las enviaría después, el mismo día).
- El estudiante mencionó tener un título de referencia interno de "Multi-Agent and Software Development" sin más detalle (aparentemente de un sistema de gestión de temas/SINFO), sin desarrollo adicional en la conversación.

## Relevancia para el proyecto

Esta reunión fija el **marco exploratorio inicial**: MARL (Multi-Agent Reinforcement Learning) + Federated Learning aplicados a entrenamiento distribuido de agentes. El survey de Gronauer y Diepold (2022) leído a continuación (ver [`survey_marl_gronauer_diepold_2022.md`](survey_marl_gronauer_diepold_2022.md)) corresponde directamente al tipo de referencia general de MARL que el asesor mencionó que enviaría.
