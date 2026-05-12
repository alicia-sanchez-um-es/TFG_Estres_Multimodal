# Detección de Estrés Multimodal (TFG)

Este repositorio contiene el código fuente y los experimentos para el Trabajo de Fin de Grado: **"Detección de Estrés mediante Modelos Multimodales y LLMs"**.

## Descripción
El objetivo del proyecto es desarrollar un sistema capaz de detectar estrés en vídeos analizando tanto la señal visual como la acústica y textual, utilizando datasets como **IEMOCAP**, **MELD** y **MSP-IMPROV**. El sistema evalúa diferentes arquitecturas de extracción (ViT, ResNet, Wav2Vec 2.0, RoBERTa, ...) y estrategias de fusión.

## Estructura del Proyecto

El proyecto sigue la siguiente organización de directorios:

- **`data/`**: Contiene la estructura de los datasets.
    - `RAW/`: (Ignorado) Datos originales de MELD e IEMOCAP.
    - `PROCESSED/`: CSVs limpios y metadatos unificados (`IEMOCAP_clean.csv`, `MELD_clean.csv`, `Multimodal_Stress_Dataset.csv`, `MSP-Improv_clean.csv`).
    - **Nota sobre características extraídas:** Debido al volumen de datos procesados (>10GB de embeddings), los tensores extraídos no se incluyen en este repositorio. Las instrucciones y el enlace de descarga al repositorio en la nube se encuentran en el archivo `data/link_to_data.txt`.
- **`notebooks/`**: Jupyter Notebooks con el pipeline ETL y el análisis exploratorio (EDA), entre otros. El orden de los mismos sigue la metodología del proyecto. 

- **`src/`**: Código fuente modular del proyecto.
    - `data/`: Scripts de carga de datos (`dataset.py`).
    - `models/`: Arquitecturas de los modelos (`adapters.py`, `fusion_strategies.py`).
    - Scripts principales: `train.py`, `evaluate.py` y `extract_visual_features.py`.
- **`resultados/`**: Reportes de métricas (F1-Score, ROC-AUC, ...) y predicciones de los modelos.
- **`figuras/`**: Gráficas, matrices de confusión y curvas de rendimiento generadas.

## Requisitos
El proyecto utiliza Python. Las principales librerías son:
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `torch` (PyTorch)
- `librosa` (Procesamiento de audio)
- `scikit-learn` (Métricas de evaluación)
- `transformers` (Hugging Face)

## Cómo empezar
1. Clonar este repositorio.
2. Descargar los datasets originales MELD e IEMOCAP y colocarlos en la carpeta `data/RAW/`.
3. Descargar las características extraídas desde el enlace proporcionado en `data/link_to_data.txt`.
4. Ejecutar los notebooks en orden numérico para replicar el preprocesamiento y EDA.
5. Utilizar los scripts de `src/` para ejecutar los entrenamientos de los modelos de fusión.

---
*Autor: SANCHEZ GOMARIZ, ALICIA*
*Tutor: RUIPEREZ VALIENTE, JOSE ANTONIO*
*Cotutor: ALBALADEJO GONZALEZ, MARIANO*
*Facultad de Informática, Grado Ciencia e Ingeniería de Datos, Universidad de Murcia*