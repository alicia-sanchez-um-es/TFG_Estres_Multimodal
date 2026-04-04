# Detección de Estrés Multimodal (TFG)

Este repositorio contiene el código fuente y los experimentos para el Trabajo de Fin de Grado: **"Detección de Estrés mediante Modelos Multimodales y LLMs"**.

## Descripción
El objetivo del proyecto es desarrollar un sistema capaz de detectar estrés en vídeos analizando tanto la señal visual como la acústica y textual, utilizando datasets como **IEMOCAP** y **MELD**. El sistema evalúa diferentes arquitecturas de extracción (ViT, ResNet, Wav2Vec 2.0, RoBERTa, ...) y estrategias de fusión.

## Estructura del Proyecto

El proyecto sigue la siguiente organización de directorios:

- **`data/`**: Contiene la estructura de los datasets.
    - `RAW/`: (Ignorado) Datos originales de MELD e IEMOCAP.
    - `PROCESSED/`: CSVs limpios y metadatos unificados (`IEMOCAP_clean.csv`, `MELD_clean.csv`, `Multimodal_Stress_Dataset.csv`).
    - **Nota sobre características extraídas:** Debido al volumen de datos procesados (>10GB de embeddings), los tensores extraídos no se incluyen en este repositorio. Las instrucciones y el enlace de descarga al repositorio en la nube se encuentran en el archivo `data/link_to_data.txt`.
- **`notebooks/`**: Jupyter Notebooks con el pipeline ETL y el análisis exploratorio (EDA), entre otros. El orden de los mismos sigue la metodología del proyecto. 
    - `1_Preprocesamiento_MELD.ipynb`: Preprocesamiento y limpieza inicial (MELD).
    - `2_Preprocesamiento_IEMOCAP.ipynb`: Preprocesamiento y limpieza inicial (IEMOCAP).
    - `3_Unificacion_y_Particion_Datos.ipynb`: Unificación y particionamiento del dataset global.
    - `4_EDA_Multimodal`: Análisis exploratorio de los datos preprocesados.
    - `5_Extraccion_Caracteristicas_Visual.ipynb`: Extracción de embeddings visuales.
    - `6_Extraccion_Caracteristicas_Audio.ipynb`: Extracción de embeddings acústicos.
    - `7_Extraccion_Caracteristicas_Texto.ipynb`: Extracción de embeddings textuales.
    - `8_Baseline_Unimodal`: Obtención de los baselines unimodales (audio, vídeo y texto).
    - `9_Seleccion_Extractor_Caracteristicas.ipynb`: Selección de los extractores de características con mayor rendimiento conjunto.
    - `10_Entrenamiento_Ajuste_Early_Fusion.ipynb`: Entrenamiento y ajuste de hiperparámetros (sobre corpus global y datasets individuales) de arquitecturas multimodales de fusión temprana. 
    - `11_Entrenamiento_Ajuste_Late_Fusion.ipynb`: Entrenamiento y ajuste de hiperparámetros (sobre corpus global y datasets individuales) de arquitecturas multimodales de fusión tardía. 
    - `12_Entrenamiento_Ajuste_Attention_Fusion.ipynb`: Entrenamiento y ajuste de hiperparámetros (sobre corpus global y datasets individuales) de arquitecturas multimodales de fusión mediante atención.
    - `13_Evaluacion_y_Seleccion_Modelos.ipynb`: Análisis comparativo de las diferentes arquitecturas, seleccionando aquellas que mayor rendimiento consiguen para la tarea, tanto en el dataset global como en los datasets individuales.
    - `14_Evaluacion_Final_Test.ipynb`: Evaluación de los modelos finales seleccionados sobre la partición de test (corpus global / datasets individuales).
    - `15_Analisis_Interpretabilidad.ipynb`: Análisis del impacto de cada modalidad en la decisión del modelo.
    - `16_Preprocesamiento_MSP-Improv_Transferencia.ipynb`: Pipeline de Preprocesamiento: limpieza inicial, armonización de etiquetas y EDA para el dataset de MSP-IMPROV.
    - `17_Evaluacion_Transferencia_Aprendizaje.ipynb`: Análisis de los resultados de Transfer Learning.
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