import argparse
import torch
import torch.nn as nn
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
# Importamos las clases que hemos creado:
from data.dataset import MultimodalStressDataset
from models.fusion_strategies import EarlyFusionBase

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación Final del Modelo Multimodal")
    parser.add_argument('--model_path', type=str, required=True, help='Ruta al archivo .pth con los mejores pesos')
    parser.add_argument('--video', type=str, required=True, choices=['resnet', 'vit', 'efficientnet'])
    parser.add_argument('--audio', type=str, required=True, choices=['wav2vec', 'mfcc'])
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--fusion', type=str, default='early')
    parser.add_argument('--audio_len', type=int, default=11)
    parser.add_argument('--video_frames', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------
    # CONFIGURACIÓN DISPOSITIVO, MAPEO DE RUTAS Y LECTURA DE DATOS
    # -------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    BASE_DIR = os.path.expanduser('/workspace')
    csv_path = os.path.join(BASE_DIR, 'Multimodal_Stress_Dataset.csv')

    df = pd.read_csv(csv_path)

    df_test = df[df['split'] == 'test']
    df_test['file_id'] = (df_test['Dialogue_ID'].astype(str) + "_" + df_test['Utterance_ID'].astype(str)).str.replace("/", "_")
    test_ids = df_test['file_id'].tolist()
    test_labels = df_test['target_stress'].tolist()

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
        # Para Wav2Vec 2.0, utilizamos el modelo base de Facebook que cuenta con una propiedad fija, donde eel extractor reduce la señal de audio que siempre genera un vector cada 20 milisegundos, es decir, 1 paso son 20 ms (0,02), por tanto calculamos de la misma manera y obtenemos que 11 segundos son 11/0,02 = 550 pasos.
        MAX_AUDIO_LEN = 550 if args.audio_len == 11 else 350 # 350 si se indica una ventana de tiempo más pequeña (de 7 segundos)
    
    # VÍDEO:
    MAX_VIDEO_FRAMES = args.video_frames
    
    if args.video == 'resnet':
        VISUAL_INPUT_DIM = 2048
    elif args.video == 'efficientnet':
        VISUAL_INPUT_DIM = 1280
    else: # vit
        VISUAL_INPUT_DIM = 768


    # -------------------------------------------
    # CARGA DE MODELO Y PESOS
    # -------------------------------------------

    if args.fusion == 'early':
        model = EarlyFusionBase(visual_dim=VISUAL_INPUT_DIM, audio_dim=AUDIO_INPUT_DIM, text_dim=768, proj_dim=512)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval() # Ponemos el modelo en modo evaluación, no hay cálculo de gradientes ni actualización de pesos (solo forward pass)

    ######## MÉTRICA I: NÚMERO DE PARÁMETROS ##############

    # MÉTRICA: NÚMERO DE PARÁMETROS
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    test_dataset = MultimodalStressDataset(
        subject_ids=test_ids, 
        labels=test_labels, 
        base_dir=BASE_DIR,
        video_folder=VIDEO_RUTA,
        audio_folder=AUDIO_RUTA,
        text_folder=TEXTO_RUTA, 
        max_audio_len=MAX_AUDIO_LEN, 
        max_video_frames=MAX_VIDEO_FRAMES
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch 1 para medir tiempo de inferencia real

    # -----------------------------------------
    # BUCLE DE EVALUACIÓN
    # -----------------------------------------
    all_preds = []
    all_probs = []
    all_labels = []
    tiempo_inferencia = []

    with torch.no_grad():
        for video_x, audio_x, text_x, labels in tqdm(test_loader):
            video_x, audio_x, text_x = video_x.to(device), audio_x.to(device), text_x.to(device)
            
            ######## MÉTRICA II: TIEMPO DE INFERENCIA ##############
            start_time = time.time()
            output = model(video_x, audio_x, text_x)
            end_time = time.time()
            
            tiempo_inferencia.append(end_time - start_time)
            
            # POC: prueba inicial con Sigmoide y BCELoss:
            # prob = output.item()

            # Con BCEWithLogitsLoss:
            prob = torch.sigmoid(output).item() # Aplicamos Sigmoide para convertir el logit en probabilidad de estrés (0 a 1)
            all_probs.append(prob)
            all_preds.append(1 if prob > 0.5 else 0)
            all_labels.append(labels.item())

    # CÁLCULO DE MÉTRICAS (accuracy, F1-Score Macro, F1-Score Weighted, AUC)
    media_tiempo_inferencia = np.mean(tiempo_inferencia) * 1000 # a milisegundos
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, all_probs)

    print("MÉTRICAS DEL MODELO EN TEST:")
    print(f"Parámetros Totales: {total_params:,}")
    print(f"Parámetros Entrenables: {trainable_params:,}")
    print(f"Tiempo medio de Inferencia: {media_tiempo_inferencia:.2f} ms / muestra")
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"Accuracy (en %): {acc*100:.2f}%")

    #  MATRIZ DE CONFUSIÓN
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Estrés', 'Estrés'], yticklabels=['No Estrés', 'Estrés'])
    plt.title(f'Matriz de Confusión - {args.fusion.upper()}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.savefig(f'Fig_5_1_1_matriz_confusion_modelo_{args.fusion}_{args.video}{args.video_frames}_{args.audio}{args.audio_len}s_{args.text}.png')


    # CURVA ROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f'Fig_5_1_2_curva_roc_modelo_{args.fusion}_{args.video}{args.video_frames}_{args.audio}{args.audio_len}s_{args.text}.png')

    # 9. INFORME FINAL (.txt)
    with open(f"reporte_final_test_modelo_{args.fusion}_{args.video}{args.video_frames}_{args.audio}{args.audio_len}s_{args.text}.txt", "w") as f:
        f.write(f"MODELO: {args.model_path}\n")
        f.write(f"Parámetros: {total_params}\n")
        f.write(f"Inferencia media: {media_tiempo_inferencia:.2f} ms\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(classification_report(all_labels, all_preds, target_names=['No Estrés', 'Estrés'], digits=4))


if __name__ == "__main__":
    main()

