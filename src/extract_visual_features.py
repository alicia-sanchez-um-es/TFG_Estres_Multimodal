# Carga previa de todas las librerías y paquetes necesarios para el script:
import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


# ------------------------------------------------------------------------
# 1. Definición del Dataset para Carga Paralela (Multiprocessing)
# ------------------------------------------------------------------------
class VideoDataset(Dataset):
    def __init__(self, df, root_meld, root_iemocap, transform=None):
        self.df = df
        self.root_meld = root_meld
        self.root_iemocap = root_iemocap
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Selección de ruta según el origen del vídeo
        if row['dataset_origin'] == 'MELD':
            video_path = os.path.join(self.root_meld, row['video_path'])
        else:
            video_path = os.path.join(self.root_iemocap, row['video_path'])
        
        # Extracción de 32 frames uniformes del vídeo
        frames = self.extract_frames(video_path)
        if self.transform and len(frames) > 0:
            frames = torch.stack([self.transform(f) for f in frames])
        
        # Retornamos el tensor de frames y el nombre formateado para el archivo .npy
        file_name = f"{row['Dialogue_ID']}_{row['Utterance_ID']}.npy".replace("/", "_")
        return frames, file_name

    def extract_frames(self, video_path, num_frames=32):
        cap = cv2.VideoCapture(video_path)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if not all_frames: return []
        indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
        return [Image.fromarray(all_frames[i]) for i in indices]

# -------------------------------------------------------------------------
# 2. Lógica Principal de Extracción
# -------------------------------------------------------------------------
def main_extraction(model_name, num_workers=8):
    """
    Función principal de extracción diseñada para ejecución paralela en HPC.
    Args:
    - model_name (str): Nombre del modelo a usar ('resnet', 'efficientnet' o 'vit').
    - num_workers (int): Número de procesos paralelos para DataLoader.
    Devuelve:
    - Guarda los embeddings extraídos en archivos .npy en el directorio correspondiente.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = f"./features_{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Configuración del Extractor según Arquitectura
    if model_name == 'resnet':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = nn.Sequential(*list(m.children())[:-1])  # Eliminamos la capa de clasificación
    elif model_name == 'efficientnet':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model = nn.Sequential(m.features, m.avgpool) # Solo queremos las features y avgpool, no la clasificación
    elif model_name == 'vit':
        # Instanciamos el modelo base original
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # Reemplazamos la cabeza (la capa que clasifica en 1000 clases) por una identidad para obtener las features directamente
        # Esto hace que el modelo devuelva directamente el vector de características (el CLS token)
        model.heads = nn.Identity()
    
    for p in model.parameters(): 
        p.requires_grad = False
    model.to(device).eval()

    # 2. Pipeline de Preprocesamiento e Inferencia
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Ejecución Paralela (DataLoader con Workers)
    df = pd.read_csv("Multimodal_Stress_Dataset.csv")
    dataset = VideoDataset(df, "/data/MELD_CLIPS", "/data", transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    with torch.no_grad():
        for frames, name in tqdm(dataloader, desc=f"Extrayendo {model_name}"):
            if frames.shape[1] == 0: 
                continue
            
            frames = frames.squeeze(0).to(device) # Shape: (32, 3, 224, 224)
            embeddings = model(frames)
            # Embeddings shape tras pasar por los modelos:
            # ResNet/EfficientNet -> (32, Dim, 1, 1) -> Necesitamos aplicar flatten
            # ViT (con Wrapper) -> (32, 768)  -> Ya está plano
            
            if model_name != 'vit':
                embeddings = embeddings.flatten(start_dim=1)
            
            np.save(os.path.join(output_dir, name[0]), embeddings.cpu().numpy())
    

# -------------------------------------------------------------------------
# PUNTO DE ENTRADA DEL SCRIPT (Main)
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # El modelo se recibe como argumento de consola, al igual que el número de workers para la ejecución paralela:
    # >>> python feature_extraction.py --model vit --workers 16
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['resnet', 'efficientnet', 'vit'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    
    main_extraction(args.model, num_workers=args.workers)