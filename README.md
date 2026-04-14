# Tesis LCD: Predicción de ingresos con EPH

Este repositorio contiene el código, notebooks y artefactos asociados al trabajo de tesis de la Licenciatura en Ciencia de Datos (LCD) orientado a la predicción de ingresos a partir de la Encuesta Permanente de Hogares (EPH).

## Objetivo
Construir y evaluar modelos de Machine Learning para predecir ingresos utilizando variables socioeconómicas de EPH, y explorar decisiones de preprocesamiento, selección de features y desempeño comparado.

## Requisitos
- Python 3.10 o superior
- Recomendado: entorno virtual


## Estructura del repositorio

* `notebooks/`: notebooks de exploración, entrenamiento y evaluación.
* `src/encuestador/`: scripts de preprocesamiento, entrenamiento y utilidades.
* `data/`:

  * `training/`: datos intermedios para entrenamiento y evaluación.
  * `info/`: metadatos y archivos auxiliares.
* `artifacts/`: modelos entrenados y salidas persistidas (puede ocupar mucho espacio).
* `figuras/`: figuras exportadas para el informe y análisis.
* `notas_metodologicas.md`: decisiones metodológicas y observaciones.

## Reproducibilidad

## Datos y artefactos

Este repositorio incluye datos y artefactos necesarios para ejecutar el flujo tal como fue usado durante la tesis. Si se desea una versión liviana, se recomienda excluir `artifacts/` y algunos archivos grandes de `data/`, y regenerarlos localmente.

## Relación con otros repositorios

Este repositorio es autónomo y está orientado a la tesis. Puede compartir ideas o fragmentos con otros proyectos de procesamiento de EPH, pero no depende de ellos.

## Autoría

* Autor: Nicolás Spisso
* Supervisión: Matías Iglesias

