import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# ARQUITECTURA MODULAR: ADAPTADORES
# ----------------------------------------------------------------------
# En esta sección, definimos adaptadores específicos para cada tipo de característica (visual, auditiva y textual).
# Cada adaptador es una LSTM + red neuronal pequeña con una capa lineal, que toma las características extraídas por los modelos preentrenados (ResNet, ViT, Wav2Vec, RoBERTa, etc.) 
# y las proyecta a un espacio latente común.
# Estos adaptadores permiten que las características de diferentes modalidades sean compatibles para la fusión multimodal posterior.
# ----------------------------------------------------------------------

# NOTA 1: Los embeddings de Vídeo y Audio entran inicialmente en formato 3D (Batch, Ventana, Características).
# Cada embedding pasa:
# 1. Por una LSTM de una sola capa oculta (en el caso solo de audio y vídeo, no en texto), que procesa la secuencia paso a paso.
# -----------> Se condensa toda la dimensión temporal en un único vector final que se toma de la última capa oculta de la LSTM (hn[-1])

#                     (Batch, Ventana, Características) --> LSTM --> (Batch, Características)
# 
# 2. Luego pasa por una red neuronal pequeña (MLP) de una sola capa lineal con batchnorm + relu + dropout que proyecta el vector al espacio latente

# NOTA 2: En el texto, con el token [CLS] directamente obtenemos un vector 2D (Batch, Características), y solo pasa por la MLP final para proyección

# NOTA 3: La memoria limitada y el problema de explosión/desvanecimiento típico de las LSTM en los adaptadores visual/audio NO afectará en nuestro caso,
# ya que la dimensión temporal en los vídeos es de 32 frames (menor a 100, por tanto es una longitud óptima para LSTM) y en audio, será de una mayor longitud (aplicando padding), pero no supondrá un problema crítico

class VisualAdapter(nn.Module):
    """Adaptador para características visuales con modelado temporal (LSTM)"""
    def __init__(self, input_dim=2048, projection_dim=512, hidden_lstm=512, dropout_prob=0.5):
        super(VisualAdapter, self).__init__()

        # PRIMERA CAPA: Red Recurrente (LSTM)
        # Lee los 32 frames (o 16) secuencialmente de una muestra y reduce la dimensionalidad original (ej. de 2048 en ResNet) a un estado oculto de (ej. 512 dimensiones)

        self.lstm = nn.LSTM(input_dim, 
                            hidden_lstm, # En el último paso de la red (último frame), nos quedaremos con el vector resultante de la CAPA OCULTA, y es el que contiene la información y memoria de toda la secuencia
                            num_layers=1, # Se fija a 1 para evitar el sobreajuste (evitar así que la red memorice los datos). manteniéndolo lo más simple posible
                            batch_first=True # IMPORTANTE!!! Ya que nuestro tensor es de tamaño (Batch, Ventana, Características) y no (Tiempo, Ventana, Características) como esperaría PyTorch
                        )

        # EJEMPLO DE RESULTADO CON VALORES PREDETERMINADOS: (Batch, 32, 2048) ---> (Batch, 512)    (Cada vídeo (32 frames) queda representado con un vector de 512 dimensiones únicamente)

        # SEGUNDA CAPA: MLP y regularización
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_lstm),  # PRIMERA CAPA: Normalización por lotes, para estabilizar el entrenamiento y acelerar la convergencia al normalizar las activaciones de la capa anterior
            nn.ReLU(), # SEGUNDA CAPA: Activación ReLU. Introduce no linealidad en la red, lo que permite modelar relaciones complejas entre las características
            nn.Dropout(dropout_prob), # TERCERA CAPA: Dropout. Ayuda a prevenir el sobreajuste al desactivar aleatoriamente un porcentaje de las neuronas durante el entrenamiento
            # CAPAS OBLIGATORIAS PARA ASEGURAR QUE el embedding resultante tenga la dimensionalidad adecuada y esté en la misma escala (media 0, desviación 1):
            nn.Linear(hidden_lstm, projection_dim), # CUARTA CAPA: Reducción final a la dimensión de proyección común (512). Esto asegura que las características visuales estén en el mismo espacio que las características auditivas y textuales para la fusión multimodal
            nn.BatchNorm1d(projection_dim) # QUINTA CAPA: Normalización por lotes final. Esto ayuda a estabilizar las activaciones de la capa de proyección y mejora la generalización del modelo. OBLIGATORIO Y ESENCIAL para que TODOS los embeddings finales (visual, audio, texto) estén en la misma escala estadística exacta (media 0, desviación 1)
        )

        #  EJEMPLO DE RESULTADO CON VALORES PREDETERMINADOS: (Batch, 512) --> (Batch, 512)

    def forward(self, x):
        # x entra con forma: (Batch, 32, input_dim)
        out, (hn, cn) = self.lstm(x)

        # hn contiene el último estado oculto tras leer toda la secuencia, lo cual representa la memoria de toda la secuencia
        # Forma original de hn: (num_layers, Batch, hidden_size)   (ej de hidden_size de 512)
        # Extraemos hn[-1] para quedarnos con el tensor plano: 
        last_hidden = hn[-1]
        return self.fc(last_hidden) # Se procesa a través de la red definida en el constructor, devolviendo las características adaptadas en el espacio común

class AudioAdapter(nn.Module):
    """
    Adaptador para características de audio (Wav2Vec o MFCC).
    Se incluye una lógica dinámica para evitar expansiones agresivas si la entrada es de baja dimensionalidad (en el caso de MFCC, que es de tan solo 15 dimensiones),
    donde en este caso último se aplica una expansión gradual.
    """
    def __init__(self, input_dim=768, projection_dim=512, hidden_lstm=512, dropout_prob=0.5):
        super(AudioAdapter, self).__init__()
        
        # MFCC -> 15 dims, Wav2Vec -> 768 dims
        # Hacemos una expansión gradual (para MFCCs) usando la memoria de la LSTM y luego lineal

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_lstm, 
            num_layers=1, # Se fija a 1 para evitar el sobreajuste (evitar así que la red memorice los datos) 
            batch_first=True                 
        )

        # EJ para MFCCs: RESULTADO: (Batch, max_audio_len, 15) --> (Batch, 512)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_lstm), # PRIMERA CAPA: Normalización por lotes
            nn.ReLU(), # SEGUNDA CAPA: Activación ReLU. Introduce no linealidad para modelar relaciones complejas entre las características de audio
            nn.Dropout(dropout_prob), # TERCERA CAPA: Dropout para prevenir el sobreajuste
            nn.Linear(hidden_lstm, projection_dim), # CUARTA CAPA: Capa lineal para expandir a 512 dimensiones
            nn.BatchNorm1d(projection_dim) # QUINTA CAPA: Asegura que el embedding final obtenido se encuentre en la misma escala estadística (media 0, desviación 1) después de la capa lineal
        )

        # EJ para MFCCs: RESULTADO: (Batch, 512) --> (Batch, 512)

    def forward(self, x):
         # x entra con forma: (Batch, Tiempo, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Extraemos hn[-1] para quedarnos con el tensor plano: 
        last_hidden = hn[-1]
        return self.fc(last_hidden) 

class TextAdapter(nn.Module):
    """
    Adaptador para características textuales.
    Al usar el token [CLS], el tensor ya es 2D (Batch, Características), por lo que usamos un MLP clásico.
    """
    def __init__(self, input_dim=768, projection_dim=512, dropout_prob=0.5):
        super(TextAdapter, self).__init__()
        # Reducción de 768 (RoBERTa, BERT, DeBERTa) 
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # EJ: RESULTADO: (Batch, 768) --> (Batch, 512)

    def forward(self, x):
        return self.net(x)