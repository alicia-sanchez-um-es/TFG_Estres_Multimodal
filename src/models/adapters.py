import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# ARQUITECTURA MODULAR: ADAPTADORES
# ----------------------------------------------------------------------
# En esta sección, definimos adaptadores específicos para cada tipo de característica (visual, auditiva y textual).
# Cada adaptador es una pequeña LSTM + red neuronal con capas lineales que toma las características extraídas por los modelos preentados (ResNet, ViT, Wav2Vec, RoBERTa, etc.) 
# y las proyecta a un espacio común latente (manifold) de 512 dimensiones.
# Estos adaptadores permiten que las características de diferentes modalidades sean compatibles para la fusión multimodal posterior.
# ----------------------------------------------------------------------

# NOTA 1: La arquitectura de las redes neuronales (adaptadores) es parecida o igual (en el caso de audio y vídeo) para mantener la consistencia en la proyección de características a un espacio común. 

# NOTA 2: Los embeddings de Vídeo y Audio entran inicialmente en formato 3D (Batch, Tiempo, Características).
# Esto pasa por una red LSTM en cada adaptador visual y acústico, que procesa la secuencia
# paso a paso. 
# Gracias a la arquitectura de la GPU del servidor DGX, este cálculo se realiza en paralelo para todas las muestras del batch, para obtener un único vector final (hn[-1]) que condensa
# toda la dinámica temporal en un tensor 2D (Batch, Características).
# El Texto se procesa directamente en capas lineales porque el token [CLS] de BERT/RoBERTa/DeBERTa ya actúa como un resumen semántico 2D, y no haría falta aplicar LSTM.

# NOTA 3: La memoria limitada y el problema de explosión/desvanecimiento típico de las LSTM a aplicar en los adaptadores visual/audio NO afectará en nuestro caso,
# ya que la dimensión temporal en los vídeos es de 32 frames (menor a 100, por tanto es una longitud óptima para LSTM) y en audio, será de una mayor longitud (aplicando padding), pero no supondrá un problema crítico. 

class VisualAdapter(nn.Module):
    """Adaptador para características visuales con modelado temporal (LSTM)"""
    def __init__(self, input_dim=2048, projection_dim=512):
        super(VisualAdapter, self).__init__()

        # PRIMERA CAPA: Red Recurrente (LSTM)
        # Lee los 32 frames secuencialmente de una muestra y reduce la dimensionalidad original (ej. de 2048 en ResNet) a un estado oculto de 512 dimensiones
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=512, # En la último paso de la red (último frame), nos quedaremos con el vector resultante de la CAPA OCULTA, que será de 512 dimensiones (512,) y es el que contiene la información y memoria de toda la secuencia
                            num_layers=1, 
                            batch_first=True # IMPORTANTE! Ya que nuestro tensor es de tamaño (Batch, Tiempo, Características) y no (Tiempo, Batch, Características) como esperaría PyTorch
                        )

        # RESULTADO: (Batch, 32, 2048) ---> (Batch, 512)    (Cada vídeo (32 frames) queda representado con un vector de 512 dimensiones únicamente)

        # SEGUNDA CAPA: MLP y regularización
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512),  # PRIMERA CAPA:  Normalización por lotes. Esto ayuda a estabilizar el entrenamiento y acelerar la convergencia al normalizar las activaciones de la capa anterior.
            nn.ReLU(), # SEGUNDA CAPA: Activación ReLU. Introduce no linealidad en la red, lo que permite modelar relaciones complejas entre las características.
            nn.Dropout(0.5), # TERCERA CAPA: Dropout. Ayuda a prevenir el sobreajuste al desactivar aleatoriamente un porcentaje de las neuronas durante el entrenamiento.
            # CAPAS OBLIGATORIAS PARA ASEGURAR QUE el embedding resultante sea de (512,) y esté en la misma escala (media 0, desviación 1):
            nn.Linear(512, projection_dim), # CUARTA CAPA: Reducción final a la dimensión de proyección común (512). Esto asegura que las características visuales estén en el mismo espacio que las características auditivas y textuales para la fusión multimodal.
            nn.BatchNorm1d(projection_dim) # QUINTA CAPA: Normalización por lotes final. Esto ayuda a estabilizar las activaciones de la capa de proyección y mejora la generalización del modelo. OBLIGATORIO Y ESENCIAL para que TODOS los embeddings finales (visual, audio, texto) estén en la misma escala estadística exacta (media 0, desviación 1)
        )

        # RESULTADO: (Batch, 512) --> (Batch, 512)

    def forward(self, x):
        # x entra con forma: (Batch, 32, input_dim)
        out, (hn, cn) = self.lstm(x)

        # hn contiene el último estado oculto tras leer toda la secuencia, lo cual representa la memoria de toda la secuencia.
        # Forma original de hn: (num_layers, Batch, hidden_size)   (hidden_size es de 512)
        # Extraemos hn[-1] para quedarnos con el tensor plano: (Batch, 512):
        last_hidden = hn[-1]
        return self.fc(last_hidden) # Se procesa a través de la red definida en el constructor, devolviendo las características adaptadas en el espacio común de 512 dimensiones.

class AudioAdapter(nn.Module):
    """
    Adaptador para características de audio (Wav2Vec o MFCC).
    Se incluye una lógica dinámica para evitar expansiones agresivas si la entrada es de baja dimensionalidad (en el caso de MFCC, que es de tan solo 15 dimensiones),
    donde en este caso último se aplica una expansión gradual.
    """
    def __init__(self, input_dim=768, projection_dim=512):
        super(AudioAdapter, self).__init__()
        
        # CASO 1: MFCC -> 15 dims
        # Hacemos una expansión gradual usando la memoria de la LSTM (15 -> 128) y luego lineal (128-> 512)
        if input_dim < 100:
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=128, num_layers=1, batch_first=True)

            # RESULTADO: (Batch, max_audio_len, 15) --> (Batch, 128)

            self.fc = nn.Sequential(
                nn.BatchNorm1d(128), # PRIMERA CAPA: Normalización por lotes. 
                nn.ReLU(), # SEGUNDA CAPA: Activación ReLU. Introduce no linealidad para modelar relaciones complejas entre las características de audio.
                nn.Dropout(0.5), # TERCERA CAPA: Dropout del 50% para prevenir el sobreajuste.
                nn.Linear(128, projection_dim), # CUARTA CAPA: Capa lineal para expandir de 128 a 512 dimensiones
                nn.BatchNorm1d(projection_dim) # QUINTA CAPA: Asegura que el embedding final obtenido se encuentre en la misma escala estadística (media 0, desviación 1) después de la capa lineal
            )

            # RESULTADO: (Batch, 128) --> (Batch, 512)
            
        # CASO 2: Wav2Vec -> 768 dims
        # Hacemos la compresión/transformación estándar primero en la LSTM (768 -> 512) y luego con una proyección lineal y regularización (512 -> 512)
        else:
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=512, num_layers=1, batch_first=True)

            # RESULTADO: (Batch, max_audio_len, 768) --> (Batch, 512)       

            self.fc = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, projection_dim), # Mantenemos la dimensión y aplicamos regularización
                nn.BatchNorm1d(projection_dim)
            )

            # RESULTADO: (Batch, 512) --> (Batch, 512)

    def forward(self, x):
         # x entra con forma: (Batch, Tiempo, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Extraemos hn[-1] para quedarnos con el tensor plano: (Batch, 128/512):
        last_hidden = hn[-1]
        return self.fc(last_hidden) 

class TextAdapter(nn.Module):
    """
    Adaptador para características textuales.
    Al usar el token [CLS], el tensor ya es 2D (Batch, Características), por lo que usamos un MLP clásico.
    """
    def __init__(self, input_dim=768, projection_dim=512):
        super(TextAdapter, self).__init__()
        # Reducción de 768 (RoBERTa, BERT, DeBERTa) a 512
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )

        # RESULTADO: (Batch, 768) --> (Batch, 512)

    def forward(self, x):
        return self.net(x)