import os
import torch
import numpy as np
from torch.utils.data import Dataset

# Para que PyTorch reconozca este archivo como "proveedor de datos oficial",
# nuestra clase MultimodalStressDataset debe heredar de torch.utils.data.Dataset, 
# y debemos implementar los métodos __len__ y __getitem__, además del constructor.

class MultimodalStressDataset(Dataset):
    """
    Dataset para cargar características multimodales (.npy) del servidor DGX.
    Permite seleccionar qué backbone visual, acústico y textual usar para cada experimento.
    """
    # CONSTRUCTOR:
    def __init__(self, subject_ids, labels, base_dir, video_folder, audio_folder, text_folder, max_audio_len, max_video_frames):
        """
        Entrada:
            subject_ids (list): IDs único (ej: '0_train_dia0_utt9.npy' ---> ID: '0_train_dia0_utt9') de cada vídeo/audio/texto.
            labels (list): Etiquetas de estrés (1/0).
            base_dir (str): Ruta raíz '~/dgx.../exp1/'
            video_folder (str):'features_resnet', 'features_vit', 'features_efficientnet'
            audio_folder (str): 'features_audio/audio_wav2vec', 'features_audio/audio_handacrafted'
            text_folder (str): EJ: 'EMBEDDINGS_TEXT_ROBERTA_64'
            max_audio_len (int): Longitud (número de pasos temporales) que cubra el 90-95% de los audios, para aplicar el padding/truncamiento y homogeneizar los tensores de audio de tamaño a variable.
            max_video_frames (int): Número de frames de acuerdo al tamaño de ventana elegido.
        """
        self.subject_ids = subject_ids
        self.labels = labels
        self.base_dir = base_dir
        self.max_audio_len = max_audio_len
        self.max_video_frames = max_video_frames
        
        # Guardamos las subcarpetas específicas del experimento actual
        self.video_folder = video_folder
        self.audio_folder = audio_folder
        self.text_folder = text_folder


    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        """
        Carga el archivo .npy de la muestra identificada con idx, y devuelve 
        dicho tensor tal cual, en el caso de audio y vídeo, y en el caso del texto devolvemos el tensor del [CLS] token que representa toda la secuencia completa 
        NOTA: Para audio, debido a que la dimensión temporal es variable ( a diferencia del vídeo (32) y del texto (32/64)),
        aplicaremos padding/truncamiento para igualar el número de pasos de tiempo en función del número de pasos que cubra el 90-95% de los audios. La longitud temporal 
        debe ser la misma ya que el DataLoader de PyTorch posterior a aplicar, requiere que la dimensión temporal de todos los elementos del batch 
        sea la misma. 
        """
        subject_id = self.subject_ids[idx]
        label = self.labels[idx]

        ## Se devuelven los embeddings tal cual los hemos obtenido (solo aplicamos padding/truncamiento al audio):



        # 1. Vídeo -----> FORMA TENSOR FINAL: ResNet (32, 2048), EfficientNet (32,1280), ViT (32, 768)
        video_path = os.path.join(self.base_dir, self.video_folder, f"{subject_id}.npy")
        video_data = np.load(video_path) 
        video_tensor = torch.tensor(video_data, dtype=torch.float32)

        ########### APLICAMOS TAMAÑO DE VENTANA ELEGIDO:
        if video_tensor.size(0) > self.max_video_frames:
        # Truncamos el vídeo si el usuario pide menos frames (en lugar de los 32 originales, 16)
        # Para ello, al igual que la estrategia original de muestreo para extraer los 32 frames de forma aleatoria uniforme,
        # de la misma manera extraemos dichos 16 frames a partir de los 32:
            indices = torch.linspace(0, video_tensor.size(0) - 1, steps=self.max_video_frames).long()
            video_tensor = video_tensor[indices] # TEMPORAL DOWNSAMPLING 



        
        # 2. AUDIO -----> FORMA TENSOR FINAL: Wav2Vec (max_audio_len, 768), MFCCs (max_audio_len, 15)
        audio_path = os.path.join(self.base_dir, self.audio_folder, f"{subject_id}.npy")
        audio_data = np.load(audio_path)
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        ######## PADDING/TRUNCAMIENTO ##########
        sec_len = audio_tensor.size(0)
        if sec_len > self.max_audio_len: # Truncamos
            audio_tensor = audio_tensor[:self.max_audio_len, :] # Truncamos
        elif sec_len < self.max_audio_len: # Padding
            padding = torch.zeros(self.max_audio_len - sec_len, audio_tensor.size(1)) # Esa diferencia de longitud pasa a ser un tensor de ceros del mismo tamaño
            audio_tensor = torch.cat((audio_tensor, padding), dim=0) # Rellenamos con ceros




        # 3. TEXTO ----> FORMA TENSOR FINAL: BERT/RoBERTa/DeBERTa (768, ) (ya que el [CLS] token devuelve el embedding que representa a toda la secuencia, equivalente a la LSTM que aplicaremos en audio/vídeo)
        text_path = os.path.join(self.base_dir, self.text_folder, f"{subject_id}.npy")
        text_data = np.load(text_path)
        # Independientemente de si es 32 o 64 tokens, el [CLS] siempre está en la posición 0 para BERT, RoBERTa y DeBERTa.
        text_tensor = torch.tensor(text_data[0, :], dtype=torch.float32)



        # La etiqueta se convierte en un tensor de tipo float32 y se le añade una dimensión extra con unsqueeze(0) para que tenga forma [1], lo que es necesario para la función de pérdida de clasificación binaria (BCE Loss) que espera una entrada de esa forma:
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0) 

        return video_tensor, audio_tensor, text_tensor, label_tensor