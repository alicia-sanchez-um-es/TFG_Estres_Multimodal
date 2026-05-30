import os
import torch
import numpy as np
from torch.utils.data import Dataset

# Para tratar nuestro dataset (tanto las fuentes individuales como el corpus unificado) como
# un dataset oficial de PyTorch (y así, por ejemplo si PyTorch requiere
# de cargar los datos, por ejemplo desde un DataLoader, llama a __len__ para saber
# el tamaño del dataset y a __getitem__(i) para obtener el dato i), se requiere
# definir obligatoriamente, a parte del __init__, los métodos __len__ y __getitem__ y
# heredar de la clase Dataset de PyTorch

class MultimodalStressDataset(Dataset):
    """
    Dataset para cargar características multimodales (también bimodales/unimodales) (.npy) del servidor DGX.
    Permite seleccionar qué backbone visual, acústico y textual usar para cada experimento,
    además de llevar a cabo el truncamiento/relleno(padding) de acuerdo a la ventana temporal de 
    audio y número de frames de vídeo.
    """
    # CONSTRUCTOR:
    def __init__(self, subject_ids, labels, df, base_dir, video_model_name = 'resnet', audio_model_name = 'wav2vec', text_model_name = 'bert64', video_folder = None, audio_folder = None, text_folder = None, max_audio_len = 550, max_video_frames = 32, modalidades = ['video','audio','texto']):
        """
        Entrada:
            subject_ids (list): IDs único (ej: '0_train_dia0_utt9.npy' ---> ID: '0_train_dia0_utt9') de cada vídeo/audio/texto.
            labels (list): Etiquetas de estrés (1/0).
            df (DataFrame): dataframe del dataset seleccionado (MELD, IEMOCAP, MSP-IMPROV).
            video_model_name (str): nombre del extractor de características visual seleccionado (RESNET, EFFICIENTNET, VIT)
            audio_model_name (str): nombre del extractor de características acústicas seleccionado (WAV2VEC, MFCC)
            text_model_name (str): nombre del extractor de características textuales (BERT, ROBERTA, DEBERTA)
            base_dir (str): Ruta raíz '~/workspace'
            video_folder (str):'EMBEDDINGS_VISUALES'
            audio_folder (str): 'EMBEDDINGS_AUDIO'
            text_folder (str): 'EMBEDDINGS_TEXTO'
            max_audio_len (int): Longitud (número de pasos temporales) para aplicar el padding/truncamiento.
            max_video_frames (int): Número de frames de acuerdo al tamaño de ventana elegido.
            modalidades (list): Lista que indica las modalidades a cargar (trimodal/bimodal/unimodal)
        """
        self.subject_ids = subject_ids
        self.labels = labels
        self.df = df 
        self.video_model_name = video_model_name.upper() # RESNET o VIT o EFFICIENTNET
        self.audio_model_name = audio_model_name.upper() # WAV2VEC o MFCC
        self.nombre_texto = text_model_name[:-2].upper() ##### ---> Extraemos el nombre del modelo: 'bert64' ---> 'BERT'
        self.ventana_texto = text_model_name[-2:] #### ----> Extraemos el tamaño de ventana de contexto: 'bert64' ---> '64'
        self.base_dir = base_dir
        self.max_audio_len = max_audio_len
        self.max_video_frames = max_video_frames
        self.video_folder = video_folder
        self.audio_folder = audio_folder
        self.text_folder = text_folder
        self.modalidades = modalidades


    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        """
        Carga el archivo .npy de la muestra identificada con idx, y devuelve 
        dicho tensor tal cual, en el caso de audio y vídeo, y en el caso del texto devolvemos el tensor del [CLS] token que representa toda la secuencia completa 
        NOTA: Para audio, debido a que la dimensión temporal es variable ( a diferencia del texto (32/64)),
        aplicaremos padding/truncamiento para igualar el número de pasos de tiempo en función del número de pasos de acuerdo al tamaño de ventana temporal seleccionado. 
        Para el vídeo, se aplicará también truncamiento en el caso de haber seleccionado un número de frames menor (16).
        """
        subject_id = self.subject_ids[idx]
        label = self.labels[idx]
        origen = self.df.iloc[idx]['dataset_origin'] #MELD o IEMOCAP o MSP-IMPROV

        #--------------------------------------------- VIDEO -----------------------------------------------------------------

        if 'video' in self.modalidades: 

            # 1. Vídeo -----> FORMA TENSOR FINAL: ResNet (32, 2048), EfficientNet (32,1280), ViT (32, 768)
            video_path = os.path.join(self.base_dir, self.video_folder, self.video_model_name, origen, f"{subject_id}.npy")

            video_data = np.load(video_path) 
            video_tensor = torch.tensor(video_data, dtype=torch.float32)

            ########### APLICAMOS TAMAÑO DE VENTANA ELEGIDO:
            if video_tensor.size(0) > self.max_video_frames:
            # Truncamos el vídeo si el usuario pide menos frames (en lugar de los 32 originales, 16)
            # Para ello, al igual que la estrategia original de muestreo para extraer los 32 frames de forma uniforme,
            # de la misma manera extraemos dichos 16 frames a partir de los 32:
                indices = torch.linspace(0, video_tensor.size(0) - 1, steps=self.max_video_frames).long()
                video_tensor = video_tensor[indices] # TEMPORAL DOWNSAMPLING 

        #-----------------------------------------------------------------------------------------------------------------------

        #--------------------------------------------- AUDIO -----------------------------------------------------------------

        
        # 2. AUDIO -----> FORMA TENSOR FINAL: Wav2Vec (max_audio_len, 768), MFCCs (max_audio_len, 15)
        if 'audio' in self.modalidades:
            audio_path = os.path.join(self.base_dir, self.audio_folder, self.audio_model_name, origen, f"{subject_id}.npy")

            audio_data = np.load(audio_path)
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
            ######## PADDING/TRUNCAMIENTO ##########
            sec_len = audio_tensor.size(0)
            if sec_len > self.max_audio_len: # Truncamos
                audio_tensor = audio_tensor[:self.max_audio_len, :] # Truncamos
            elif sec_len < self.max_audio_len: # Padding
                padding = torch.zeros(self.max_audio_len - sec_len, audio_tensor.size(1)) # Esa diferencia de longitud pasa a ser un tensor de ceros del mismo tamaño
                audio_tensor = torch.cat((audio_tensor, padding), dim=0) # Rellenamos con ceros

        #--------------------------------------------------------------------------------------------------------------------

        #--------------------------------------------- TEXTO -----------------------------------------------------------------


        # 3. TEXTO ----> FORMA TENSOR FINAL: BERT/RoBERTa/DeBERTa (768, ) (ya que el [CLS] token devuelve el embedding que representa a toda la secuencia, equivalente a la LSTM que aplicaremos en audio/vídeo)
        if 'texto' in self.modalidades: 
            path_particion = f"{origen}_{self.ventana_texto}"

            text_path = os.path.join(self.base_dir, self.text_folder,self.nombre_texto, path_particion, f"{subject_id}.npy")
            text_data = np.load(text_path)
            # Independientemente de si es 32 o 64 tokens, el [CLS] siempre está en la posición 0 para BERT, RoBERTa y DeBERTa.
            text_tensor = torch.tensor(text_data[0, :], dtype=torch.float32)

        #--------------------------------------------------------------------------------------------------------------------------

        # La etiqueta se convierte en un tensor de tipo float32 y se le añade una dimensión extra con unsqueeze(0) para que tenga forma [1], lo que es necesario para la función de pérdida que espera una entrada de esa forma:
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        # torch.tensor(label, dtype=torch.float32) --> tensor escalar con 0 dimensiones: (ej: label = 1 --> tensor(1.)) 
        # con .unsqueeze(0) añadimos una dimensión ---> ej: tensor(1.) --> tensor([1.]) con forma [1]

        # ESTO ES ASÍ porque BCEWithLogistLoss requiere de que la predicción y la etiqueta tengan la misma forma:
        # ej: prediction = tensor([0.8]) con forma [1] y label_tensor = tensor([1.]) con forma [1] 

        sample = {'label': label_tensor}
        
        # Añadimos solo las modalidades que hemos cargado:
        if 'video' in self.modalidades:
            sample['video'] = video_tensor
        if 'audio' in self.modalidades:
            sample['audio'] = audio_tensor
        if 'texto' in self.modalidades:
            sample['texto'] = text_tensor

        return sample