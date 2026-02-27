import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
import os
import pandas as pd
# Importamos las clases que hemos creado:
from data.dataset import MultimodalStressDataset
from models.fusion_strategies import EarlyFusionBase, LateFusionBase

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento Multimodal para Detección de Estrés")

    # Argumentos de entrada por terminal directamente: 

    # 1. BACKBONES:
    parser.add_argument('--video', type=str, required=True, choices=['resnet', 'vit', 'efficientnet'], 
                        help='Backbone visual a utilizar')
    parser.add_argument('--audio', type=str, required=True, choices=['wav2vec', 'mfcc'], 
                        help='Backbone acústico a utilizar')
    parser.add_argument('--text', type=str, required=True, choices=['roberta64', 'bert64', 'deberta64', 'bert32', 'roberta32', 'deberta32'], 
                        help='Backbone textual a utilizar y su ventana de tokens')
    
    # 2. ESTRATEGIA DE FUSIÓN:
    parser.add_argument('--fusion', type=str, default='early', choices=['early', 'late', 'attention'],
                        help='Estrategia de fusión (early, late o attention)')
    
    # 3. VENTANAS TEMPORALES DE AUDIO Y VIDEO:
    parser.add_argument('--audio_len', type=int, default=11, choices=[11, 7],
                        help='Ventana temporal del audio en segundos (11 o 7)')
    parser.add_argument('--video_frames', type=int, default=32, choices=[32, 16],
                        help='Número de frames de vídeo a procesar (32 o 16)')
    
    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------
    # MAPEO DE RUTAS
    # -------------------------------------------
    mapeo_rutas = {
        'video': {
            'resnet': 'features_resnet',
            'vit': 'features_vit',
            'efficientnet': 'features_efficientnet'
        },
        'audio': {
            'wav2vec': 'features_audio_COMPLETO/audio_wav2vec',
            'mfcc': 'features_audio_COMPLETO/audio_handcrafted'
        },
        'text': {
            'roberta64': 'EMBEDDINGS_TEXT_ROBERTA_64',
            'bert64': 'EMBEDDINGS_TEXT_BERT_64',
            'deberta64': 'EMBEDDINGS_TEXT_DEBERTA_64',
            'roberta32': 'EMBEDDINGS_TEXT_ROBERTA_32',
            'bert32': 'EMBEDDINGS_TEXT_BERT_32',
            'deberta32': 'EMBEDDINGS_TEXT_DEBERTA_32'
        }
    }

    VIDEO_RUTA = mapeo_rutas['video'][args.video]
    AUDIO_RUTA = mapeo_rutas['audio'][args.audio]
    TEXTO_RUTA = mapeo_rutas['text'][args.text]

    #------------------------------------------------------------------
    # CÁLCULO DINÁMICO DE PASOS DE TIEMPO (TIME STEPS) Y MAX_AUDIO_LEN
    # -----------------------------------------------------------------
    # AUDIO:
    if args.audio == 'mfcc':
        AUDIO_INPUT_DIM = 15
        # En MFCCs, en la extracción indicamos sr=16000 y hop_length=512, esto es que 1 paso son 512/16000 = 0,032 segundos (32 ms), por tanto 11 segundos son 11/0,032 = 343,27 pasos, redondeándolo a 350 pasos. 
        MAX_AUDIO_LEN = 350 if args.audio_len == 11 else 220 # 220 si se indica una ventana de tiempo más pequeña (de 7 segundos)
    else: # wav2vec
        AUDIO_INPUT_DIM = 768 
        # Para Wav2Vec 2.0, utilizamos el modelo base de Facebook que cuenta con una propiedad fija, donde el extractor reduce la señal de audio que siempre genera un vector cada 20 milisegundos, es decir, 1 paso son 20 ms (0,02), por tanto calculamos de la misma manera y obtenemos que 11 segundos son 11/0,02 = 550 pasos.
        MAX_AUDIO_LEN = 550 if args.audio_len == 11 else 350 # 350 si se indica una ventana de tiempo más pequeña (de 7 segundos)
    
    # VÍDEO:
    MAX_VIDEO_FRAMES = args.video_frames
    
    if args.video == 'resnet':
        VISUAL_INPUT_DIM = 2048
    elif args.video == 'efficientnet':
        VISUAL_INPUT_DIM = 1280
    else: # vit
        VISUAL_INPUT_DIM = 768

    

    #------------------------------------------------------------------
    # CONFIGURACIÓN DEL ENTORNO
    # -----------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hiperparámetros básicos (estos posteriormente se optimizarán con un ajuste de hiperparámetros más exhaustivo):
    BATCH_SIZE = 32 
    LEARNING_RATE = 1e-4 # Con este valor inicial de learning rate, el modelo comienza a aprender sin hacer cambios demasiado bruscos en los pesos, para una mayor estabilidad del entrenamiento.
    EPOCHS = 20 # Debido al tamaño pequeño del dataset, no debemos emplear muchas epochs para evitar el sobreajuste

    #------------------------------------------------------------------
    # LECTURA DE DATOS (en el servidor DGX)
    # -----------------------------------------------------------------
           
    BASE_DIR = os.path.expanduser('/workspace')
    csv_path = os.path.join(BASE_DIR, 'Multimodal_Stress_Dataset.csv')

    df = pd.read_csv(csv_path)

    # Creamos una columna temporal llamada 'file_id' en df con el nombre exacto de los archivos .npy (sin la extensión)
    # FORMATO: "Dialogue_ID_Utterance_ID" reemplanzando cualquieer barra por guión bajo
    df['file_id'] = (df['Dialogue_ID'].astype(str) + "_" + df['Utterance_ID'].astype(str)).str.replace("/", "_")

    # FILTRAMOS los datos de TRAIN (Columna 'split' == 'train') y los de Validación ('dev'):
    df_train = df[df['split']=='train']
    train_ids = df_train['file_id'].tolist()
    train_labels = df_train['target_stress'].tolist()

    df_val = df[df['split']=='dev']
    val_ids = df_val['file_id'].tolist()
    val_labels = df_val['target_stress'].tolist()
    

    print('DETECCIÓN DE ESTRÉS: ')
    print(f"--> Fusión: {args.fusion.upper()}")
    print(f"--> Vídeo: {args.video} ({MAX_VIDEO_FRAMES} frames) -> {VIDEO_RUTA}")
    print(f"--> Audio: {args.audio} ({args.audio_len}s -> {MAX_AUDIO_LEN} pasos) -> {AUDIO_RUTA}")
    print(f"--> Texto: {args.text} -> {TEXTO_RUTA}")

    # -------------------- DATASET Y DATALOADER TRAIN ------------------

    # Instanciamos el Dataset con nuestra clase creada: 
    train_dataset = MultimodalStressDataset(
        subject_ids=train_ids,
        labels= train_labels,
        base_dir= BASE_DIR,  # La ruta hacia nuestro workspace en el servidor
        video_folder= VIDEO_RUTA,
        audio_folder= AUDIO_RUTA,
        text_folder= TEXTO_RUTA,
        max_audio_len=MAX_AUDIO_LEN,
        max_video_frames = MAX_VIDEO_FRAMES
    )

    # DataLoader permite cargar los datos en lotes y barajarlos durante el entrenamiento:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # shuffle = True para mezclar los datos en cada epoch y evitar que el modelo aprenda un orden específico, lo que mejora la generalización

    # -------------------- DATASET Y DATALOADER VAL ----------------
    val_dataset = MultimodalStressDataset(
        subject_ids=val_ids,
        labels= val_labels,
        base_dir=BASE_DIR,  
        video_folder=VIDEO_RUTA,
        audio_folder=AUDIO_RUTA,
        text_folder= TEXTO_RUTA,
        max_audio_len = MAX_AUDIO_LEN,
        max_video_frames = MAX_VIDEO_FRAMES
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # Aquí, shuffle = False porque no queremos mezclar los datos de validación, ya que esto no afecta al rendimiento del modelo



    # ---------------------------------------------------------
    # INSTANCIACIÓN DEL MODELO Y MECANISMO EARLY STOPPING
    # ---------------------------------------------------------
    if args.fusion == 'early':
        model = EarlyFusionBase(visual_dim=VISUAL_INPUT_DIM, audio_dim=AUDIO_INPUT_DIM, text_dim=768, proj_dim=512)
    elif args.fusion == 'late':
        model = LateFusionBase(visual_dim=VISUAL_INPUT_DIM, audio_dim=AUDIO_INPUT_DIM, text_dim=768, proj_dim=512)
    # else:
    #     model = AttentionFusionBase(visual_dim=VISUAL_INPUT_DIM, audio_dim=AUDIO_INPUT_DIM, text_dim=768, proj_dim=512)
    
    model = model.to(device)

    # FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR

    # 1. POC: Prueba inicial con BCELoss, pero se cambió a BCEWithLogitsLoss para mayor estabilidad numérica al trabajar con logits en la capa final, especialmente dado el desbalanceo del dataset:
    # Usamos BCELoss ya que nuestro problema es una clasificación binaria (Estrés vs No Estrés). Esta función de pérdida compara las probabilidades predichas por el modelo con las etiquetas reales (0 o 1) y penaliza las predicciones incorrectas.
    # criterion = nn.BCELoss() 

    # 2. CAMBIO A BCEWithLogitsLoss: Esta función de pérdida combina una capa sigmoide con la función de pérdida de entropía cruzada, lo que es más estable numéricamente para problemas de clasificación binaria, especialmente con datasets desbalanceados como el nuestro. Al usar esta función, podemos dejar la capa final del modelo sin activación (logits) y la función de pérdida se encargará de aplicar la sigmoide internamente
    # Calculamos dinámicamente cuántos hay de cada clase en Train:
    num_negativos = train_labels.count(0)
    num_positivos = train_labels.count(1)

    # Fórmula del peso: (Nº muestras clase mayoritaria) / (Nº muestras clase minoritaria)
    pos_weight = num_negativos / num_positivos if num_positivos > 0 else 1.0 # Evitamos división por cero, si no hay positivos, asignamos un peso de 1 (sin ponderación)
    # EJ: pos_weight de 3,4 indica que la clase positiva (estrés) es 3,4 veces menos frecuente que la clase negativa (no estrés), por lo que el modelo penalizará 3,4 veces más los errores en la clase positiva para ayudar a aprender mejor a identificarla
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)) # Asignamos el peso a la clase positiva (estrés) para penalizar más los errores en esa clase y ayudar al modelo a aprender mejor a identificarla, dado el desbalanceo del dataset
    # El parámetro pos_weight sirve para asignar peso a la clase positiva (minoritaria)

    # AdamW como optimizador para evitar el sobreajuste
    optimizer = optim.AdamW(model.parameters(),  
                            lr=LEARNING_RATE, 
                            weight_decay=1e-2) # 0,01 de weight decay para añadir regularización L2 y ayudar a prevenir el sobreajuste, penalizando el aumento o crecimiento de los pesos de la red
    
    # CONFIGURACIÓN EARLY STOPPING:
    best_val_f1 = 0.0
    paciencia_limite = 5 # Establecemos 5 epochs como máximo que aguantamos sin mejorar antes de parar
    contador_paciencia = 0
    nombre_modelo = f"modelo_estres_{args.fusion}_{args.video}{args.video_frames}_{args.audio}{args.audio_len}s_{args.text}.pth" # EJ: modelo_estres_early_resnet32_wav2vec11s_roberta64.pth

    # ---------------------------------------------------------
    # BUCLE DE ENTRENAMIENTO (EPOCHS)
    # ---------------------------------------------------------

    for epoch in range(EPOCHS):
        model.train() # Ponemos el modelo en modo entrenamiento (activamos el Dropout)
        running_loss = 0.0

        #==============================
        # TRAIN
        #==============================
    
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for video_x, audio_x, text_x, labels in progress_bar:
            # Enviamos los datos del lote a la GPU:
            video_x = video_x.to(device)
            audio_x = audio_x.to(device)
            text_x = text_x.to(device)
            labels = labels.to(device)

            # Reseteamos los gradientes (obligatorio en PyTorch en cada iteración)
            optimizer.zero_grad()
            # Pasamos los datos por el modelo -- Forward Propagation 
            predictions = model(video_x, audio_x, text_x)
            # Cálculo del Error (Loss)
            loss = criterion(predictions, labels)
            # Backward Propagation - Calculamos los gradientes de la función de pérdida con respecto a los pesos del modelo
            loss.backward()
            # Actualizamos los pesos de la red
            optimizer.step()
            # Estadísticas visuales
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
            
        #==============================
        # VALIDACIÓN
        #==============================
        model.eval() # Apagamos el dropout para la validación
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Listas temporales para el F1-Score de esta epoch:
        epoch_labels = []
        epoch_preds = []

        with torch.no_grad(): # No calculamos gradientes durante la validación para ahorrar memoria y acelerar el proceso
            for video_x, audio_x, text_x, labels in val_loader:
                video_x = video_x.to(device)
                audio_x = audio_x.to(device)
                text_x = text_x.to(device)
                labels = labels.to(device)

                predictions = model(video_x, audio_x, text_x)
                # Calculamos el error de validación:
                loss = criterion(predictions, labels)
                val_loss += loss.item()

                # POC: como en la prueba inicial tenemos la Sigmoide, directamente aplicamos esta línea:
                # Calculamos cuántas acertó (predicciones > 0.5 se consideran clase 1, y < 0.5 se consideran clase 0)
                # predicted_labels = (predictions > 0.5).float()
                # 
                # Con BCEWithLogitsLoss, al tener logits a la salida, aplicamos Sigmoide ahora:
                probabilidades = torch.sigmoid(predictions)
                predicted_labels = (probabilidades > 0.5).float() 
                correct_predictions += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)

                epoch_labels.extend(labels.cpu().numpy())
                epoch_preds.extend(predicted_labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_samples 

        # Calculamos el F1-Score Macro para penalizar el desbalanceo:
        val_f1 = f1_score(epoch_labels, epoch_preds, average='macro')

        # Mostramos resultados de la epoch:
        print(f"Epoch {epoch+1}. Train Loss: {train_loss:.4f}. Val Loss: {val_loss:.4f}. Val Acc: {val_accuracy*100:.2f}%. Val F1 Macro: {val_f1:.4f}\n")

        # ---------------------------------------------
        # EARLY STOPPING Y CHECKPOINT
        # ---------------------------------------------
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # =========================================================
            # GUARDADO DEL MODELO 
            # =========================================================
            torch.save(model.state_dict(), nombre_modelo) # Guardamos el "cerebro" (los pesos) de la red en el directorio actual (./)
            print(f"Nuevo mejor modelo: (F1: {best_val_f1:.4f}) -> Guardando pesos...")
            contador_paciencia = 0 # Reseteamos el contador porque ha mejorado
        else:
            contador_paciencia += 1
            print(f"Sin mejora en F1-Score. Límite: {contador_paciencia}/{paciencia_limite}")
            
            if contador_paciencia >= paciencia_limite:
                print(f"\nEARLY STOPPING El modelo no ha mejorado en {paciencia_limite} epochs.")
                print(f"Deteniendo el entrenamiento en la Epoch {epoch+1} para evitar sobreajuste.")
                break # Rompemos el bucle

    print(f"ENTRENAMIENTO FINALIZADO. Pesos del modelo guardados localmente en: ./{nombre_modelo}")

if __name__ == "__main__":
    main()