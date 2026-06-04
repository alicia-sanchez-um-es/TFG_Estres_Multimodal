import os
import pandas as pd
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score
# Importamos las clases que hemos creado:
from data.dataset import MultimodalStressDataset
from models.unimodal_classifier import UnimodalClassifier
from models.adapters import VisualAdapter, AudioAdapter, TextAdapter
from models.fusion_strategies import EarlyFusion, LateFusion, AttentionFusion

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo (unimodal, bimodal o trimodal) para la detección de estrés")

    # Argumentos de entrada por terminal directamente: 

    # 1. SELECCIÓN DEL DOMINIO (GLOBAL vs INDIVIDUAL)
    parser.add_argument('--train_dataset', type=str, default='global', choices=['global', 'MELD', 'IEMOCAP'], 
                        help='Corpus de entrenamiento: global (ambos), MELD o IEMOCAP')

    # 2. BACKBONES:
    parser.add_argument('--video', type=str, default=None, choices=['resnet', 'vit', 'efficientnet'], 
                        help='Backbone visual a utilizar (no poner nada si no se usa vídeo)')
    parser.add_argument('--audio', type=str, default=None, choices=['wav2vec', 'mfcc'], 
                        help='Backbone acústico a utilizar (no poner nada si no se usa audio)')
    parser.add_argument('--text', type=str, default=None, choices=['roberta64', 'bert64', 'deberta64', 'bert32', 'roberta32', 'deberta32'], 
                        help='Backbone textual a utilizar y su ventana de tokens (no poner nada si no se usa texto)')
    
    # 3. ESTRATEGIA DE FUSIÓN:
    parser.add_argument('--fusion', type=str, default='early', choices=['early', 'late', 'attention'],
                        help='Estrategia de fusión (early, late o attention)')
    # Para Late Fusion, se indica también la técnica de fusión a utilizar:
    parser.add_argument('--late_mode', type=str, default='promedio',
                    choices=['voto', 'promedio', 'logistica'],
                    help='Técnica de combinación de decisiones para Late Fusion')
    
    # 4. HIPERPARÁMETROS A AJUSTAR:

    # ---> VENTANAS TEMPORALES DE AUDIO Y VIDEO:
    parser.add_argument('--audio_len', type=int, default=11, choices=[11, 7],
                        help='Ventana temporal del audio en segundos (11 o 7)')
    parser.add_argument('--video_frames', type=int, default=32, choices=[32, 16],
                        help='Número de frames de vídeo a procesar (32 o 16)')
    
    # ---> CONFIGURACIÓN DE ARQUITECTURA:
    parser.add_argument('--proj_dim', type=int, default=512, help='Dimensión de proyección')
    parser.add_argument('--hidden_mlp', type=int, default=128, help='Dimensión oculta del clasificador final')

    # ---> ENTRENAMIENTO Y REGULARIZACIÓN
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Probabilidad de Dropout')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--epochs', type=int, default=20, help='Número máximo de epochs')
    parser.add_argument('--patience', type=int, default=5, help='Paciencia para Early Stopping')
    parser.add_argument('--pos_weight_mult', type=float, default=1.0, help='Multiplicador para el peso de la clase positiva')

    # 5. TRANSFER-LEARNING: Entrenamiento con corpus MSP-IMPROV/MELD/IEMOCAP:
    parser.add_argument('--transfer_learning', type=str, default=None, choices=['MELD_MSP-IMPROV', 'IEMOCAP_MSP-IMPROV'],
                        help='Activa la evaluación de transferencia cruzada entrenando con la combinación indicada. Sobrescribe a --train_dataset. Solo aplicable para arquitecturas trimodales.')
    
    return parser.parse_args()


def main():
    args = parse_args()

    mods = [] # Aquí guardamos las modalidades seleccionadas por el usuario
    if args.video is not None: 
        mods.append('video')
    if args.audio is not None: 
        mods.append('audio')
    if args.text is not None: 
        mods.append('texto')

    if len(mods) == 0:
        raise ValueError("Debes especificar al menos un backbone (--video, --audio o --text) para evaluar el modelo.")


    #------------------------------------------------------------------
    # CONFIGURACIÓN DEL ENTORNO
    # -----------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hiperparámetros básicos:
    BATCH_SIZE = args.batch_size 
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs

    # -------------------------------------------
    # MAPEO DE RUTAS
    # -------------------------------------------
    mapeo_rutas = {
        'video': 'EMBEDDINGS_VISUAL',
        'audio': 'EMBEDDINGS_AUDIO',
        'text': 'EMBEDDINGS_TEXTO'
        }

    VIDEO_RUTA = mapeo_rutas['video']
    AUDIO_RUTA = mapeo_rutas['audio']
    TEXTO_RUTA = mapeo_rutas['text']

    #------------------------------------------------------------------
    # CÁLCULO DINÁMICO DATOS EN SERVIDOR DGX y TAMAÑO VENTANA
    # -----------------------------------------------------------------

    # -------------- 1. CARGA DE DATOS train y dev -------------------

    BASE_DIR = os.path.expanduser('/workspace')
    csv_path = os.path.join(BASE_DIR, 'Multimodal_Stress_Dataset.csv')
    df = pd.read_csv(csv_path)
    # Creamos una columna temporal llamada 'file_id' en df con el nombre exacto de los archivos .npy (sin la extensión)
    # FORMATO: "(dataset_origin)_(Utterance_ID)" reemplanzando cualquier barra por guión bajo
    df['file_id'] = (df['dataset_origin'].astype(str) + "_" + df['Utterance_ID'].astype(str)).str.replace("/", "_") 

        # -------------------------- VALIDACIÓN TRANSFER LEARNING: MSP-IMPROV/MELD/IEMOCAP ------------------

    if args.transfer_learning is not None: 
        csv_path = os.path.join(BASE_DIR, 'MSP-Improv_clean.csv')
        df_msp = pd.read_csv(csv_path)
        df_msp['dataset_origin'] = 'MSP-IMPROV' # <- Ahora sí añadimos la columna que indique el origen del dataset, ya que se concatenará con MELD o IEMOCAP a continuación

        if args.transfer_learning == 'MELD_MSP-IMPROV':
            df = df[df['dataset_origin'] == 'MELD']

        elif args.transfer_learning == 'IEMOCAP_MSP-IMPROV':
            df = df[df['dataset_origin'] == 'IEMOCAP']
        
        df = pd.concat([df, df_msp], ignore_index=True)
        

        # ------------------------- MELD / IEMOCAP / GLOBAL (MELD + IEMOCAP) --------------------------
    else:
        # Si se ha seleccionado los datasets individuales, filtramos el dataset global por la columna 'dataset_origin':
        if args.train_dataset != 'global':
            df = df[df['dataset_origin'] == args.train_dataset]

    
    # FILTRAMOS los datos de TRAIN (columna 'split' == 'train') y los de validación ('dev'):

    df_train = df[df['split']=='train']
    train_ids = df_train['file_id'].tolist()
    train_labels = df_train['target_stress'].tolist()

    df_val = df[df['split']=='dev']
    val_ids = df_val['file_id'].tolist()
    val_labels = df_val['target_stress'].tolist()

    # -----------------------------------------------------------------

    # VIDEO:

    VISUAL_INPUT_DIM = 2048 if args.video == 'resnet' else (1280 if args.video == 'efficientnet' else 768)
    MAX_VIDEO_FRAMES = args.video_frames

    # AUDIO:

    AUDIO_INPUT_DIM = 15 if args.audio == 'mfcc' else 768
    # Para hacer la comparativa justa entre MFCCs y Wav2Vec 2.0, ambos presentan frecuencia de muestreo de 16kHz (sr=16000) y un hop_length de 320. Por tanto, extraen 1 vector de características cada 320/16000 = 0,02 (20 ms). Con esa configuración, para 11s, se limita dicha ventana temporal en MAX_AUDIO_LEN = 550 (11/0,02 = 550 vectores de características), mientras que para 7s el tamaño de la ventana queda con 350 vectores extraídos (7/0,02 = 350)
    MAX_AUDIO_LEN = 550 if args.audio_len == 11 else 350 


    #------------------------------------------------------------------
    # CREACIÓN MULTIMODAL DATASET Y DATALOADER
    # -----------------------------------------------------------------

    train_dataset = MultimodalStressDataset(
        subject_ids=train_ids, labels=train_labels, df=df_train,
        video_model_name=args.video if args.video else 'none',
        audio_model_name=args.audio if args.audio else 'none',
        text_model_name=args.text if args.text else 'none00', # indicamos none00 para que no falle el slicing 
        base_dir=BASE_DIR, video_folder=VIDEO_RUTA, audio_folder=AUDIO_RUTA, text_folder=TEXTO_RUTA,
        max_audio_len=MAX_AUDIO_LEN, max_video_frames=MAX_VIDEO_FRAMES,
        modalidades=mods
    )

    val_dataset = MultimodalStressDataset(
        subject_ids=val_ids, labels=val_labels, df=df_val,
        video_model_name=args.video if args.video else 'none',
        audio_model_name=args.audio if args.audio else 'none',
        text_model_name=args.text if args.text else 'none00',
        base_dir=BASE_DIR, video_folder=VIDEO_RUTA, audio_folder=AUDIO_RUTA, text_folder=TEXTO_RUTA,
        max_audio_len=MAX_AUDIO_LEN, max_video_frames=MAX_VIDEO_FRAMES,
        modalidades=mods
    )


    # DataLoader permite cargar los datos en lotes y barajarlos durante el entrenamiento:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # shuffle = True para mezclar los datos en cada epoch y evitar que el modelo aprenda un orden específico, lo que mejora la generalización
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # Aquí, shuffle = False porque no queremos mezclar los datos de validación, ya que esto no afecta al rendimiento del modelo


    # ---------------------------------------------------------
    # INSTANCIACIÓN DE LOS MÓDULOS, MENSAJE AL USUARIO, 
    # DEFINICIÓN DE NOMBRES DE ARCHIVOS Y MECANISMO EARLY STOPPING
    # ---------------------------------------------------------

    # -------------------- UNIMODAL -------------------------
    # Para los experimentos unimodales, almacenamos todo el historial en un único fichero .json para mayor comodidad: 
    if len(mods) == 1 :
        # Instanciamos el adaptador correcto de acuerdo a la modalidad:
        if args.video is not None: 
            print(f"ENTRENAMIENTO BASELINE:\nModalidad: VIDEO\nBackbone: {args.video.upper()}")
            hidden_lstm = max(args.proj_dim, VISUAL_INPUT_DIM // 4)
            model = UnimodalClassifier(VisualAdapter(input_dim=VISUAL_INPUT_DIM, projection_dim = args.proj_dim, hidden_lstm=hidden_lstm, dropout_prob=args.dropout), proj_dim=args.proj_dim, hidden_mlp=args.hidden_mlp, dropout_prob=args.dropout).to(device)
            nombre_modelo = f"pesos_baseline_unimodal_video_{args.video}_ventana{MAX_VIDEO_FRAMES}_pdim{args.proj_dim}_hmlp{args.hidden_mlp}_do{args.dropout}_lr{LEARNING_RATE}.pth"
        elif args.audio is not None: 
            print(f"ENTRENAMIENTO BASELINE:\nModalidad: AUDIO\nBackbone: {args.audio.upper()}")
            hidden_lstm = max(args.proj_dim, AUDIO_INPUT_DIM // 4)
            model = UnimodalClassifier(AudioAdapter(input_dim=AUDIO_INPUT_DIM, projection_dim = args.proj_dim, hidden_lstm=hidden_lstm, dropout_prob=args.dropout), proj_dim=args.proj_dim, hidden_mlp=args.hidden_mlp, dropout_prob=args.dropout).to(device)
            nombre_modelo = f"pesos_baseline_unimodal_audio_{args.audio}_ventana{args.audio_len}_pdim{args.proj_dim}_hmlp{args.hidden_mlp}_do{args.dropout}_lr{LEARNING_RATE}.pth"
        elif args.text is not None: 
            print(f"ENTRENAMIENTO BASELINE:\nModalidad: TEXTO\nBackbone: {args.text.upper()}")
            model = UnimodalClassifier(TextAdapter(input_dim=768, projection_dim = args.proj_dim, dropout_prob=args.dropout), proj_dim=args.proj_dim, hidden_mlp=args.hidden_mlp, dropout_prob=args.dropout).to(device)
            nombre_modelo = f"pesos_baseline_unimodal_texto_{args.text}_ventana{args.text[-2:]}_pdim{args.proj_dim}_hmlp{args.hidden_mlp}_do{args.dropout}_lr{LEARNING_RATE}.pth"
        else: 
            raise ValueError("Modalidad incorrecta. Usa: video, audio o texto")
        
    # -------------------- BIMODAL/TRIMODAL -------------------------

    else: # len(mods) > 1
        # Mostramos por pantalla el tipo de entrenamiento iniciado por el usuario:
        print('DETECCIÓN DE ESTRÉS: ') 
        print(f"--> Fusión: {args.fusion.upper()}")
    
        if mods == ['video', 'audio']:
            print(f"--> Vídeo: {args.video} ({MAX_VIDEO_FRAMES} frames) -> {VIDEO_RUTA}")
            print(f"--> Audio: {args.audio} ({args.audio_len}s -> {MAX_AUDIO_LEN} pasos) -> {AUDIO_RUTA}")
            mods_str = f"{args.video}{args.video_frames}_{args.audio}{args.audio_len}s"
            nombre_dataset_archivo = args.train_dataset
        elif mods == ['video', 'texto']:
            print(f"--> Vídeo: {args.video} ({MAX_VIDEO_FRAMES} frames) -> {VIDEO_RUTA}")
            print(f"--> Texto: {args.text} -> {TEXTO_RUTA}")
            mods_str = f"{args.video}{args.video_frames}_{args.text}"
            nombre_dataset_archivo = args.train_dataset
        elif mods == ['audio', 'texto']:
            print(f"--> Audio: {args.audio} ({args.audio_len}s -> {MAX_AUDIO_LEN} pasos) -> {AUDIO_RUTA}")
            print(f"--> Texto: {args.text} -> {TEXTO_RUTA}")
            mods_str = f"{args.audio}{args.audio_len}s_{args.text}"
            nombre_dataset_archivo = args.train_dataset

        else : # len(mods) == 3 
            print(f"--> Vídeo: {args.video} ({MAX_VIDEO_FRAMES} frames) -> {VIDEO_RUTA}")
            print(f"--> Audio: {args.audio} ({args.audio_len}s -> {MAX_AUDIO_LEN} pasos) -> {AUDIO_RUTA}")
            print(f"--> Texto: {args.text} -> {TEXTO_RUTA}")
            mods_str = f"{args.video}{args.video_frames}_{args.audio}{args.audio_len}s_{args.text}"
            nombre_dataset_archivo = args.transfer_learning if args.transfer_learning is not None else args.train_dataset # Si se aplicase transfer_learning solo se aplica al caso trimodal

        late_str = f"_{args.late_mode}" if args.fusion == 'late' else ""
        sufijo = f"{nombre_dataset_archivo}_{args.fusion}{late_str}_{mods_str}_p{args.proj_dim}_h{args.hidden_mlp}_lr{args.lr}_do{args.dropout}"
        nombre_modelo = f"pesos_modelo_estres_{sufijo}.pth"
        nombre_historial = f"historial_estres_{sufijo}.json"

        # ---------- TÉCNICAS DE FUSIÓN ----------------

        if args.fusion == 'early':
            model = EarlyFusion(visual_dim=VISUAL_INPUT_DIM, audio_dim=AUDIO_INPUT_DIM, text_dim=768, proj_dim=args.proj_dim, hidden_mlp=args.hidden_mlp, dropout_prob=args.dropout, modalidades=mods)
        elif args.fusion == 'late':
            model = LateFusion(visual_dim=VISUAL_INPUT_DIM, audio_dim=AUDIO_INPUT_DIM, text_dim=768, proj_dim=args.proj_dim, hidden_mlp=args.hidden_mlp, dropout_prob=args.dropout, fusion_mode = args.late_mode, modalidades = mods)
        elif args.fusion == 'attention':
            model = AttentionFusion(visual_dim=VISUAL_INPUT_DIM, audio_dim=AUDIO_INPUT_DIM, text_dim=768, proj_dim=args.proj_dim, hidden_mlp=args.hidden_mlp, dropout_prob=args.dropout, modalidades=mods)
        else: 
            raise ValueError("Estrategia no válida. Usa: early, late, attention")
    
    model = model.to(device)

    # FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR

    # 1. POC: Prueba inicial con BCELoss, pero se cambió a BCEWithLogitsLoss para mayor estabilidad numérica al trabajar con logits en la capa final, especialmente dado el desbalanceo del dataset:
    # funcion_perdida = nn.BCELoss() 

    # 2. CAMBIO A BCEWithLogitsLoss: Esta función de pérdida combina una capa sigmoide con la función de pérdida de entropía cruzada, lo que es más estable numéricamente para problemas de clasificación binaria, especialmente con datasets desbalanceados como el nuestro. Al usar esta función, podemos dejar la capa final del modelo sin activación (logits) y la función de pérdida se encargará de aplicar la sigmoide internamente
    # Calculamos cuántos hay de cada clase en Train (este parámetro se fija en train para validación y test, para evitar así la fuga de datos)
    num_negativos = train_labels.count(0)
    num_positivos = train_labels.count(1)

    # Fórmula del peso: (Nº muestras clase mayoritaria) / (Nº muestras clase minoritaria)
    pos_weight = num_negativos / num_positivos if num_positivos > 0 else 1.0 # Evitamos división por cero, si no hay positivos, asignamos un peso de 1 (sin ponderación)

    # Le aplicamos el multiplicador indicado para aplicar un mayor o menor peso (aunque en nuestras pruebas de experimentación lo vamos a dejar fijo a 1.0):
    pos_weight *= args.pos_weight_mult

    # EJ: Un pos_weight de 1.4 indica que hay aproximadamente 1.4 veces más muestras negativas que positivas (por cada muestra positiva, hay 1.4 muestras negativas), por lo que el modelo penalizará 1.4 veces más los errores en la clase positiva
    funcion_perdida = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)) 
    # El parámetro pos_weight sirve para asignar peso a la clase positiva (minoritaria)

    # AdamW como optimizador para evitar el sobreajuste
    optimizer = optim.AdamW(model.parameters(),  
                            lr=LEARNING_RATE, 
                            weight_decay=args.weight_decay) 

    
    # CONFIGURACIÓN EARLY STOPPING:
    best_val_f1 = 0.0
    paciencia_limite = args.patience
    contador_paciencia = 0

    # DICCIONARIO para guardar el historial de métricas:
    history = {
        'train_loss': [],
        'val_loss':[],
        'val_f1':[],
        'val_recall_estres':[]
    }

    # ---------------------------------------------------------
    # BUCLE DE ENTRENAMIENTO (EPOCHS)
    # ---------------------------------------------------------

    for epoch in range(EPOCHS):
        model.train() # Ponemos el modelo en modo entrenamiento 
        running_loss = 0.0

        #==============================
        # TRAIN
        #==============================
    
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress_bar :
            labels = batch['label'].to(device) # Ya tiene la forma [Batch, 1]
            video_x = batch['video'].to(device) if 'video' in batch else None
            audio_x = batch['audio'].to(device) if 'audio' in batch else None
            text_x = batch['texto'].to(device) if 'texto' in batch else None

            # Reseteamos los gradientes (obligatorio en PyTorch en cada iteración)
            optimizer.zero_grad()
            # Pasamos los datos por el modelo -- Forward Propagation 
            if len(mods) == 1:
                unico_input = video_x if video_x is not None else (audio_x if audio_x is not None else text_x)
                predictions = model(unico_input)
            else :
                predictions = model(video_x=video_x, audio_x=audio_x, text_x=text_x) # ya el modelo se encarga de procesar solo las modalidades que le hayamos indicado
            # Cálculo del error (loss)
            loss = funcion_perdida(predictions, labels)
            # Backward propagation: calculamos los gradientes de la función de pérdida con respecto a los pesos del modelo
            loss.backward()
            # Actualizamos los pesos de la red
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
            
        #==============================
        # VALIDACIÓN
        #==============================
        model.eval() # Apagamos el dropout para la validación
        val_loss = 0.0

        epoch_labels = []
        epoch_preds = []

        with torch.no_grad(): # No calculamos gradientes durante la validación 
            for batch in val_loader :
                labels = batch['label'].to(device)
                video_x = batch['video'].to(device) if 'video' in batch else None
                audio_x = batch['audio'].to(device) if 'audio' in batch else None
                text_x = batch['texto'].to(device) if 'texto' in batch else None

                if len(mods) == 1:
                    unico_input = video_x if video_x is not None else (audio_x if audio_x is not None else text_x)
                    predictions = model(unico_input)

                else :
                    predictions = model(video_x=video_x, audio_x=audio_x, text_x=text_x)
                # Calculamos el error de validación:
                loss = funcion_perdida(predictions, labels)
                val_loss += loss.item()

                # POC: como en la prueba inicial tenemos la sigmoide, directamente aplicamos esta línea:
                # Calculamos cuántas acertó (predicciones > 0.5 se consideran clase 1, y < 0.5 se consideran clase 0)
                # predicted_labels = (predictions > 0.5).float()
                # 
                # Con BCEWithLogitsLoss, al tener logits a la salida, aplicamos sigmoide ahora:
                probabilidades = torch.sigmoid(predictions)
                predicted_labels = (probabilidades > 0.5).float() 

                epoch_labels.extend(labels.cpu().numpy().flatten().tolist()) #Utilizamos .flatten() para aplanar a [Batch] y evitar errores con scikit-learn
                epoch_preds.extend(predicted_labels.cpu().numpy().flatten().tolist())

        val_loss = val_loss / len(val_loader)

        # Calculamos el F1-Score Macro para penalizar el desbalanceo:
        val_f1 = f1_score(epoch_labels, epoch_preds, average='macro')

        # Calculamos el recall de la clase estrés (pos_label=1):
        val_recall_estres = recall_score(epoch_labels, epoch_preds, pos_label=1, zero_division=0)

        # Mostramos resultados de la epoch:
        print(f"Epoch {epoch+1}. Train Loss: {train_loss:.4f}. Val Loss: {val_loss:.4f}. Val F1 Macro: {val_f1:.4f}. Val Recall Estrés: {val_recall_estres:.4f}\n")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_recall_estres'].append(val_recall_estres)

        # ---------------------------------------------
        # EARLY STOPPING Y CHECKPOINT
        # ---------------------------------------------
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # GUARDADO DEL MODELO:
            torch.save(model.state_dict(), nombre_modelo) # Guardamos los pesos de la red en el directorio actual (./)
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

    if len(mods) == 1 : 
        nombre_historial_individual = f"historial_{nombre_modelo.replace('.pth', '.json')}"
        with open(nombre_historial_individual, 'w') as f:
            json.dump(history, f)
        print(f"Historial de entrenamiento guardado en: ./{nombre_historial_individual}")

    else : 
        with open(nombre_historial, 'w') as f:
            json.dump(history, f)
        print(f"Historial de entrenamiento guardado en: ./{nombre_historial}")

if __name__ == "__main__":
    main()