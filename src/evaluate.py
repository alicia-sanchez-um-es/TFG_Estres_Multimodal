import os
import pandas as pd
import numpy as np
import argparse
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
# Importamos las clases que hemos creado:
from data.dataset import MultimodalStressDataset
from models.fusion_strategies import EarlyFusion, LateFusion, AttentionFusion
from models.unimodal_classifier import UnimodalClassifier
from models.adapters import VisualAdapter, AudioAdapter, TextAdapter

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Se esperaba un valor booleano (True/False).')
    

def parse_args():
    # Argumentos de entrada por terminal directamente: 
    parser = argparse.ArgumentParser(description="Evaluación del Modelo Multimodal")

    parser.add_argument('--model_path', type=str, required=True, help='Ruta al archivo .pth con los pesos')
    parser.add_argument('--eval_dataset', type=str, default='global', choices=['global', 'MELD', 'IEMOCAP'], 
                        help='Corpus a evaluar: global (ambos), MELD o IEMOCAP')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test'], help='Conjunto a evaluar: dev o test')

    # 1. BACKBONES:
    parser.add_argument('--video', type=str, default=None, choices=['resnet', 'vit', 'efficientnet'], 
                        help='Backbone visual a utilizar (no poner nada si no se usa vídeo)')
    parser.add_argument('--audio', type=str, default=None, choices=['wav2vec', 'mfcc'], 
                        help='Backbone acústico a utilizar (no poner nada si no se usa audio)')
    parser.add_argument('--text', type=str, default=None, choices=['roberta64', 'bert64', 'deberta64', 'bert32', 'roberta32', 'deberta32'], 
                        help='Backbone textual a utilizar y su ventana de tokens (no poner nada si no se usa texto)')
    
    # 2. ESTRATEGIA DE FUSIÓN:
    parser.add_argument('--fusion', type=str, default='early', choices=['early', 'late', 'attention'], help='Estrategia de fusión (early, late o attention)')
    # Para Late Fusion, se indica también la técnica de fusión a utilizar:
    parser.add_argument('--late_mode', type=str, default='promedio',
                    choices=['voto', 'promedio', 'logistica'],
                    help='Técnica de combinación de decisiones para Late Fusion')

    # 3. HIPERPARÁMETROS:

    # ---> VENTANAS TEMPORALES DE AUDIO Y VIDEO:
    parser.add_argument('--audio_len', type=int, default=11, choices=[11, 7],help='Ventana temporal del audio en segundos (11 o 7)')
    parser.add_argument('--video_frames', type=int, default=32, choices=[32, 16],help='Número de frames de vídeo a procesar (32 o 16)')

    # ---> CONFIGURACIÓN DE ARQUITECTURA:
    parser.add_argument('--proj_dim', type=int, default=512, help='Dimensión de proyección')
    parser.add_argument('--hidden_mlp', type=int, default=128, help='Dimensión oculta del clasificador final')
    
    # ---> REGULARIZACIÓN
    parser.add_argument('--dropout', type=float, default=0.5, help='Probabilidad de Dropout')

    # ---> ANÁLISIS DE SESGO DE GÉNERO:
    parser.add_argument('--gender_bias', type=str2bool, default=False, help='Pon True para realizar y graficar el análisis de sesgo de género para la arquitectura multimodal de video, audio y texto solo.')

    # ---> ANÁLISIS DE INTERPRETABILIDAD:
    parser.add_argument('--interp_atencion', type=str2bool, default=False, help='Pon True para extraer y guardar los pesos de atención en un CSV para la arquitectura multimodal de video, audio y texto solo.')

    # 4. TRANSFER-LEARNING:
    parser.add_argument('--transfer_learning', type=str, default=None, choices=['MSP-IMPROV', 'IEMOCAP', 'MELD'],
                        help='Activa la evaluación de transferencia de aprendizaje para evaluar al modelo sobre el dataset completo indicado. Sobrescribe a --eval_dataset. Solo aplicable para arquitecturas trimodales.')
    
    # 5. ESTUDIO DE ABLACIÓN:
    parser.add_argument('--ablation_study', nargs='+', default=[], choices=['video', 'audio', 'texto'], 
                        help='Modalidades a apagar para el estudio de ablación (ej: --ablation_study video texto). Solo aplicable para arquitecturas multimodales (bi o tri).')

    return parser.parse_args()
    

def main():
    args = parse_args()

    # -------------------------------------------
    # CONFIGURACIÓN DISPOSITIVO, MAPEO DE RUTAS Y LECTURA DE DATOS
    # -------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mods = []
    if args.video is not None: mods.append('video')
    if args.audio is not None: mods.append('audio')
    if args.text is not None: mods.append('texto')

    if len(mods) == 0:
        raise ValueError("Debes especificar al menos un backbone (--video, --audio o --text) para evaluar el modelo.")

    mapeo_rutas = {
        'video': 'EMBEDDINGS_VISUAL',
        'audio': 'EMBEDDINGS_AUDIO',
        'text': 'EMBEDDINGS_TEXTO'
        }

    VIDEO_RUTA = mapeo_rutas['video']
    AUDIO_RUTA = mapeo_rutas['audio']
    TEXTO_RUTA = mapeo_rutas['text']

    BASE_DIR = os.path.expanduser('/workspace')

    #------------------------------------------------------------------
    # CARGA DE LOS DATOS / TRANSFER LEARNING
    # -----------------------------------------------------------------

    if args.transfer_learning == 'MSP-IMPROV':
        csv_path = os.path.join(BASE_DIR, 'MSP-Improv_clean.csv')
        df_eval = pd.read_csv(csv_path)
        df_eval['dataset_origin'] = 'MSP-IMPROV' # <- Añadimos la columna que indique el origen del dataset, ya que el CSV de MSP-IMPROV no la tiene pero el de MELD/IEMOCAP sí, y necesitamos esta columna para cargar los datos correctamente en el Dataset 
    
    else:
        csv_path = os.path.join(BASE_DIR, 'Multimodal_Stress_Dataset.csv')
        df_eval = pd.read_csv(csv_path)

        if args.transfer_learning is not None:
            df_eval = df_eval[df_eval['dataset_origin'] == args.transfer_learning]
        else:
            # Si NO es Transfer Learning, filtramos por el split (dev/test):
            df_eval = df_eval[df_eval['split'] == args.split]
            # Y si no es global, filtramos también por el corpus específico:
            if args.eval_dataset != 'global':
                df_eval = df_eval[df_eval['dataset_origin'] == args.eval_dataset]
        
        df_eval = df_eval.reset_index(drop=True)
        df_eval['file_id'] = (df_eval['dataset_origin'].astype(str) + "_" + df_eval['Utterance_ID'].astype(str)).str.replace("/", "_")

    eval_ids = df_eval['file_id'].tolist()
    eval_labels = df_eval['target_stress'].tolist()

    #------------------------------------------------------------------
    # CÁLCULO DINÁMICO DE PASOS DE TIEMPO (TIME STEPS) Y MAX_AUDIO_LEN
    # -----------------------------------------------------------------
    # AUDIO:
    if args.audio == 'mfcc':
        AUDIO_INPUT_DIM = 15
    else: # wav2vec
        AUDIO_INPUT_DIM = 768

    MAX_AUDIO_LEN = 550 if args.audio_len == 11 else 350  
        
    
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

    # -------------------- UNIMODAL -------------------------

    if len(mods) == 1 :
        # Instanciamos el adaptador correcto de acuerdo a la modalidad:
        if args.video is not None: 
            modalidad = 'video'
            hidden_lstm = max(args.proj_dim, VISUAL_INPUT_DIM // 4)
            model = UnimodalClassifier(VisualAdapter(input_dim=VISUAL_INPUT_DIM, projection_dim = args.proj_dim, hidden_lstm=hidden_lstm, dropout_prob=args.dropout), proj_dim=args.proj_dim, hidden_mlp=args.hidden_mlp, dropout_prob=args.dropout).to(device)
        elif args.audio is not None: 
            modalidad = 'audio'
            hidden_lstm = max(args.proj_dim, AUDIO_INPUT_DIM // 4)
            model = UnimodalClassifier(AudioAdapter(input_dim=AUDIO_INPUT_DIM, projection_dim = args.proj_dim, hidden_lstm=hidden_lstm, dropout_prob=args.dropout), proj_dim=args.proj_dim, hidden_mlp=args.hidden_mlp, dropout_prob=args.dropout).to(device)
        elif args.text is not None: 
            modalidad = 'texto'
            model = UnimodalClassifier(TextAdapter(input_dim=768, projection_dim = args.proj_dim, dropout_prob=args.dropout), proj_dim=args.proj_dim, hidden_mlp=args.hidden_mlp, dropout_prob=args.dropout).to(device)
        else: 
            raise ValueError("Modalidad incorrecta. Usa: video, audio o texto")
        
    # -------------------- BIMODAL/TRIMODAL -------------------------

    if len(mods) == 2 or len(mods) == 3:
        if args.fusion == 'early':
            model = EarlyFusion(visual_dim=VISUAL_INPUT_DIM, audio_dim=AUDIO_INPUT_DIM, text_dim=768, proj_dim=args.proj_dim, hidden_mlp=args.hidden_mlp, dropout_prob=args.dropout, modalidades=mods)
        elif args.fusion == 'late':
            model = LateFusion(visual_dim=VISUAL_INPUT_DIM, audio_dim=AUDIO_INPUT_DIM, text_dim=768, proj_dim=args.proj_dim, hidden_mlp=args.hidden_mlp, dropout_prob=args.dropout, fusion_mode = args.late_mode, modalidades = mods)
        elif args.fusion == 'attention':
            model = AttentionFusion(visual_dim=VISUAL_INPUT_DIM, audio_dim=AUDIO_INPUT_DIM, text_dim=768, proj_dim=args.proj_dim, hidden_mlp=args.hidden_mlp, dropout_prob=args.dropout, modalidades=mods)
        else: 
            raise ValueError("Estrategia no válida. Usa: early, late, attention")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval() # Ponemos el modelo en modo evaluación, no hay cálculo de gradientes ni actualización de pesos (solo forward pass)

    ######## MÉTRICA: NÚMERO DE PARÁMETROS ##############

    # MÉTRICA: NÚMERO DE PARÁMETROS
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    eval_dataset = MultimodalStressDataset(
        subject_ids=eval_ids, 
        labels=eval_labels,
        df = df_eval,
        video_model_name = args.video if args.video else 'none',
        audio_model_name=args.audio if args.audio else 'none',
        text_model_name=args.text if args.text else 'none00',
        base_dir=BASE_DIR,
        video_folder=VIDEO_RUTA,
        audio_folder=AUDIO_RUTA,
        text_folder=TEXTO_RUTA, 
        max_audio_len=MAX_AUDIO_LEN, 
        max_video_frames=MAX_VIDEO_FRAMES,
        modalidades = mods
    )
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False) # Batch 1 para medir tiempo de inferencia real

    # -----------------------------------------
    # BUCLE DE EVALUACIÓN
    # -----------------------------------------
    all_preds = []
    all_probs = []
    all_labels = []
    tiempo_inferencia = []

    all_attention_weights = [] # Guardamos aquí los pesos de atención

    nombre_base = os.path.basename(args.model_path).replace('.pth', '')

    if len(args.ablation_study) > 0 and len(mods) > 1:
        sufijo_ablacion = "_sin_" + "_".join(args.ablation_study)
        nombre_base += sufijo_ablacion

    backbone_unimodal = args.video if args.video is not None else (args.audio if args.audio is not None else args.text)

    if args.transfer_learning is not None:
        # Ej: "_transfer_learning_IEMOCAP"
        sufijo_final = f"transfer_learning_{args.transfer_learning}"
    else:
        if len(mods) == 1:
            sufijo_final = args.split # Ej: "test"
        else:
            sufijo_final = f"{args.eval_dataset}_{args.split}" # Ej: "global_test"

    if len(mods) == 1: 
        nombre_mc = f'Fig_matriz_confusion_baseline_unimodal_{modalidad.upper()}_{backbone_unimodal}_{sufijo_final}.png'
        nombre_roc = f'Fig_curva_roc_baseline_unimodal_{modalidad.upper()}_{backbone_unimodal}_{sufijo_final}.png'
        nombre_reporte = f"reporte_final_baseline_unimodal_{modalidad.upper()}_{backbone_unimodal}_{sufijo_final}.txt"
    else : 
        nombre_mc = f"Fig_matriz_{nombre_base}_{sufijo_final}.png"
        nombre_roc = f"Fig_roc_{nombre_base}_{sufijo_final}.png"
        nombre_reporte = f"reporte_{nombre_base}_{sufijo_final}.txt"

    with torch.no_grad():
        for batch in tqdm(eval_loader) :
            labels = batch['label'].to(device) # Ya tiene la forma [Batch, 1]
            video_x = batch['video'].to(device) if 'video' in batch else None
            audio_x = batch['audio'].to(device) if 'audio' in batch else None
            text_x = batch['texto'].to(device) if 'texto' in batch else None

            #-----------------------------------------------------------------------------------
            # ESTUDIO DE ABLACIÓN 
            #-------------------------------------------------------------------------------------
            if len(args.ablation_study) > 0 and len(mods) > 1:
                if 'video' in args.ablation_study and video_x is not None:
                    video_x = torch.zeros_like(video_x)
                if 'audio' in args.ablation_study and audio_x is not None:
                    audio_x = torch.zeros_like(audio_x)
                if 'texto' in args.ablation_study and text_x is not None:
                    text_x = torch.zeros_like(text_x)
        
            
            ######## MÉTRICA: TIEMPO DE INFERENCIA ##############
            start_time = time.time()
            # Si el modelo es el de atención y el usuario explícitamente indicó "si" para el análisis de interpretabilidad:
            if len(mods) == 1:
                unico_input = video_x if video_x is not None else (audio_x if audio_x is not None else text_x)
                output = model(unico_input)
            elif args.fusion == "attention" and args.interp_atencion and len(mods) == 3:
                output, attn_weights = model(video_x= video_x, audio_x=audio_x, text_x=text_x, return_attention=True)
                # Guardamos los pesos, con squeeze quitamos las dimensiones vacías para que quede [Vídeo, Audio, Texto]
                all_attention_weights.append(attn_weights.squeeze().cpu().numpy().tolist())
            else :
                output = model(video_x=video_x, audio_x=audio_x, text_x=text_x)
            end_time = time.time()
            
            tiempo_inferencia.append(end_time - start_time)
            
            # POC: prueba inicial con Sigmoide y BCELoss:
            # prob = output.item()

            # Con BCEWithLogitsLoss:
            probs = torch.sigmoid(output) # Aplicamos Sigmoide para convertir el logit en probabilidad de estrés (0 a 1)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy().flatten().tolist()) #Utilizamos .flatten() para aplanar a [Batch] y evitar errores con scikit-learn
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_probs.extend(probs.cpu().numpy().flatten().tolist())

    ######## MÉTRICAS : Accuracy, Balanced Accuracy, F1-Score Macro, F1-Score Weighted, AUC ##############

    media_tiempo_inferencia = np.mean(tiempo_inferencia) * 1000 # a milisegundos
    acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, all_probs)

    print(f"MÉTRICAS DEL MODELO EN {args.split.upper()}:")
    print(f"Parámetros Totales: {total_params:,}")
    print(f"Parámetros Entrenables: {trainable_params:,}")
    print(f"Tiempo medio de Inferencia: {media_tiempo_inferencia:.2f} ms / muestra")
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"Accuracy (en %): {acc*100:.2f}%")
    print(f"Balanced Accuracy (en %): {balanced_acc*100:.2f}%")


    ######## MÉTRICA : MATRIZ DE CONFUSIÓN ##############

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Estrés', 'Estrés'], yticklabels=['No Estrés', 'Estrés'])
    plt.title(f'Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.savefig(nombre_mc)
    plt.close()

    # GRÁFICO CURVA ROC:
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(nombre_roc)
    plt.close()

    # INFORME FINAL (.txt)
    with open(nombre_reporte, "w") as f:
        f.write(f"MODELO: {args.model_path}\n")
        f.write(f"Parámetros: {total_params}\n")
        f.write(f"Inferencia media: {media_tiempo_inferencia:.2f} ms\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(classification_report(all_labels, all_preds, target_names=['No Estrés', 'Estrés'], digits=4))

    # ----------------------------------------------------------
    # ANÁLISIS DE INTERPRETABILIDAD 
    # ---------------------------------------------------------

    ############################# ATENCIÓN ###############################

    if args.fusion == 'attention' and args.interp_atencion and len(mods) == 3:
        df_atencion = pd.DataFrame(all_attention_weights, columns=['Peso_Video', 'Peso_Audio', 'Peso_Texto'])
        df_atencion['Label_Real'] = all_labels
        df_atencion['Prediccion'] = all_preds
        # Guardamos en CSV:
        nombre_csv = f"interpretabilidad_atencion_{nombre_base}_{sufijo_final}.csv"
        df_atencion.to_csv(nombre_csv, index=False)
        # Mostramos el promedio global de los pesos y una vista previa de las primeras 5 muestras:
        print("\nRESUMEN DE INTERPRETABILIDAD (Atención)")
        promedios = df_atencion[['Peso_Video', 'Peso_Audio', 'Peso_Texto']].mean()
        print(f" Vídeo : {promedios['Peso_Video']*100:.2f}%")
        print(f" Audio : {promedios['Peso_Audio']*100:.2f}%")
        print(f" Texto : {promedios['Peso_Texto']*100:.2f}%")
        print("\nVista previa de las 5 primeras muestras (CSV):")
        print(df_atencion.head().to_string(index=False))

    # ----------------------------------------------------------
    # ANÁLISIS DE SESGO DE GÉNERO (FAIRNESS)
    # ---------------------------------------------------------
    if args.gender_bias and len(mods) == 3:
        print("\n ANÁLISIS DE SESGO DE GÉNERO: ")

        df_bias = df_eval.copy()
        
        # Verificamos que exista la columna gender, para mayor seguridad:
        if 'gender' not in df_bias.columns:
            print("  [ERROR] No se encontró la columna 'gender' en el dataset.")
        else:
            df_bias['true_label'] = all_labels
            df_bias['pred_label'] = all_preds
            df_bias['pred_prob'] = all_probs

            # Filtramos solo Masculino (M) y Femenino (F) 
            df_mf = df_bias[df_bias['gender'].isin(['M', 'F'])]

            def calcular_metricas_grupo(df_grupo):
                y_true = df_grupo['true_label']
                y_pred = df_grupo['pred_label']
                
                # Matriz de confusión para sacar TPR (Recall) y FPR
                # Usamos labels=[0, 1] por si algún subgrupo solo tiene una clase
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0 # True Positive Rate (Recall)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 # False Positive Rate
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                
                return {'F1-Macro': f1, 'TPR': tpr, 'FPR': fpr, 'N': len(y_true)}

            metricas_F = calcular_metricas_grupo(df_mf[df_mf['gender'] == 'F'])
            metricas_M = calcular_metricas_grupo(df_mf[df_mf['gender'] == 'M'])

            print(f" FEMENINO  (N={metricas_F['N']}): F1-Macro: {metricas_F['F1-Macro']:.4f} | Recall: {metricas_F['TPR']:.4f} | FPR: {metricas_F['FPR']:.4f}")
            print(f" MASCULINO (N={metricas_M['N']}): F1-Macro: {metricas_M['F1-Macro']:.4f} | Recall: {metricas_M['TPR']:.4f} | FPR: {metricas_M['FPR']:.4f}")

            # --- Visualización con Seaborn ---
            plot_data = pd.DataFrame({
                'Género': ['Femenino', 'Femenino', 'Femenino', 'Masculino', 'Masculino', 'Masculino'],
                'Métrica': ['F1-Macro', 'TPR (Recall)', 'FPR (Falsas Alarmas)', 'F1-Macro', 'TPR (Recall)', 'FPR (Falsas Alarmas)'],
                'Valor': [metricas_F['F1-Macro'], metricas_F['TPR'], metricas_F['FPR'],
                          metricas_M['F1-Macro'], metricas_M['TPR'], metricas_M['FPR']]
            })

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='Métrica', y='Valor', hue='Género', data=plot_data, palette={'Femenino': "#0dcd26", 'Masculino': '#1f77b4'})
            plt.ylim(0, 1.05)
            plt.ylabel('Puntuación')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Añadimos el valor numérico encima de cada barra
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9, color='#4a4a4a')

            plt.legend(title='Género', loc='upper right')
            plt.tight_layout()
            
            nombre_grafica = f"Fig_sesgo_genero_{nombre_base}_{sufijo_final}.png"
            plt.savefig(nombre_grafica, dpi=300)
            plt.close()

    


if __name__ == "__main__":
    main()

