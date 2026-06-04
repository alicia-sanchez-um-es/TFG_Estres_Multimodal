# Detección de estrés multimodal en entornos conversacionales (TFG)

Este repositorio contiene el código fuente y los experimentos para el Trabajo de Fin de Grado: **"Detección de estrés multimodal en entornos conversacionales"**.

## Descripción
El objetivo principal de este proyecto es evaluar el impacto de la multimodalidad y el entrenamiento multidominio en la detección automática de estrés en entornos conversacionales, analizando si la integración de señales de audio, vídeo y texto de diversos corpus mejora el rendimiento y la capacidad de generalización frente a modelos entrenados en dominios y modalidades aisladas. Para ello, se han empleado datasets como **IEMOCAP**, **MELD** y **MSP-IMPROV**, y se han probado diversos modelos preentrenados para la extracción de características (ViT, ResNet, Wav2Vec 2.0, RoBERTa, ...), junto con diferentes estrategias de fusión.

## Estructura del Proyecto

El proyecto sigue la siguiente organización de directorios:

- **`data/`**: No se incluye debido a las restricciones de acceso de los datasets de **IEMOCAP** y **MSP-IMPROV**.
- **`notebooks/`**: Jupyter Notebooks con el pipeline ETL y el análisis exploratorio (EDA), entre otros. El orden de los mismos sigue la metodología del proyecto. 

- **`src/`**: Código fuente modular del proyecto.
    - `data/`: Scripts de carga de datos (`dataset.py`).
    - `models/`: Arquitecturas de los modelos (`adapters.py`, `fusion_strategies.py`, `unimodal_classifier.py`).
    - Scripts principales: `train.py`, `evaluate.py`.
- **`resultados/`**: Reportes de métricas (F1-Score, ROC-AUC, ...) y predicciones de los modelos.
- **`figuras/`**: Gráficas generadas presentes en la memoria del proyecto.

## Datasets 
- **MELD**: Disponible públicamente. Descarga en https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
- **IEMOCAP**: Acceso bajo solicitud en https://sail.usc.edu/iemocap/
- **MSP-IMPROV**: Acceso bajo solicitud en https://www.lab-msp.com/MSP/MSP-Improv.html

## Requisitos
El proyecto utiliza Python. Las principales librerías son:
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `torch` (PyTorch)
- `librosa` (Procesamiento de audio)
- `scikit-learn` (Métricas de evaluación)
- `transformers` (Hugging Face)

## Cómo empezar
1. Clonar este repositorio.
2. Descargar los datasets originales MELD, IEMOCAP y MSP-IMPROV y colocarlos en la carpeta `data/RAW/`.
3. Ejecutar los notebooks en orden numérico para replicar el preprocesamiento y EDA.
4. Utilizar los scripts de `src/` para ejecutar los entrenamientos de los modelos de fusión.

---
*Autor: SANCHEZ GOMARIZ, ALICIA*
*Tutor: RUIPEREZ VALIENTE, JOSE ANTONIO*
*Cotutor: ALBALADEJO GONZALEZ, MARIANO*
*Facultad de Informática, Grado Ciencia e Ingeniería de Datos, Universidad de Murcia*
