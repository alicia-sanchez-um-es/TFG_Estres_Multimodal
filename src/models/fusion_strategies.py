import torch
import torch.nn as nn
# Importamos los adaptadores creados desde adapters.py:
from .adapters import VisualAdapter, AudioAdapter, TextAdapter

# ------------ PRIMERA ESTRATEGIA DE FUSIÓN: EARLY FUSION ----------
# En esta estrategia, se concatenan los embeddings proyectados a un espacio común antes de la clasificación.
class EarlyFusionBase(nn.Module):
    """
    Arquitectura de Fusión Temprana (Early Fusion).
    Concatena las representaciones unimodales a nivel de características.
    Esta clase lo que hace es:
    - Instanciar los adaptadores para cada modalidad (visual, audio, texto) que proyectan las características a un espacio común.
    - Concatenar las características proyectadas de las tres modalidades.
    - Pasar la representación multimodal fusionada a través de un MLP para la clasificación final.

    Devuelve:
    - Un valor entre 0 y 1 que representa la probabilidad de que el vídeo esté etiquetado como Estrés (1) o No Estrés (0).
    """
    def __init__(self, visual_dim=2048, audio_dim=768, text_dim=768, proj_dim=512, hidden_lstm=512, hidden_mlp=512, dropout_prob=0.5, lstm_layers=1):
        super(EarlyFusionBase, self).__init__()

        # Este constructor sirve de "pegamento" para los adaptadores creados en adapters.py y el MLP FINAL para clasificación definido a continuación
        
        # 1. Instanciamos los Adaptadores 
        self.visual_adapter = VisualAdapter(input_dim=visual_dim, projection_dim=proj_dim, hidden_lstm=hidden_lstm, dropout_prob=dropout_prob, lstm_layers=lstm_layers)
        self.audio_adapter = AudioAdapter(input_dim=audio_dim, projection_dim=proj_dim, hidden_lstm=hidden_lstm, dropout_prob=dropout_prob, lstm_layers=lstm_layers)
        self.text_adapter = TextAdapter(input_dim=text_dim, projection_dim=proj_dim, dropout_prob=dropout_prob)
        
        # 2. El Clasificador Final (MLP)
        # La entrada es la suma de los tres espacios proyectados
        fusion_dim = proj_dim * 3
        
        ## Arquitectura del MLP:
        ## - Capa 1 LINEAL: Reducción de proj_dim*3 a `hidden_mlp`` dimensiones. Primer paso hacia una reducción gradual de la dimensionalidad, permite que la red aprenda a condensar la información multimodal.
        ##     --> ReLU: Para introducir no linealidad, clave para que la red aprenda relaciones no lineales entre las características multimodales.
        ##     --> Dropout: Para prevenir el sobreajuste, dado el tamaño relativamente pequeño del dataset, debemos incluir regularización.
        ## - Capa 2 LINEAL: Reducción de `hidden_mlp` a 128 dimensiones. Se continúa la reducción gradual.
        ##     --> ReLU: Para mantener la capacidad de modelar relaciones complejas.
        ##     --> Dropout: Del 50% para seguir previniendo el sobreajuste.
        ## - Capa Final LINEAL: Reducción de 128 a 1 dimensión. Esta neurona final devuelve un valor bruto (logit).
        ##     --> Sigmoid: Para convertir el logit en una probabilidad entre 0 y 1, que es lo que necesitamos para la clasificación binaria de estrés/no estrés

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_mlp, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            # Capa final: 1 neurona que devuelve un valor bruto (logit)
            nn.Linear(128, 1)
            
            # POC inicial, se probó con una capa final sigmoide como prueba inicial, obteniendo resultados bajos para nuestro dataset desbalanceado
            # Sigmoide para convertir ese valor en probabilidad de estrés (0 a 1)
            # nn.Sigmoid() 

            # CAMBIO A LOGITS: Se decidió dejar la capa final sin activación (logits) para utilizar BCEWithLogitsLoss, que combina la función de pérdida de entropía cruzada con una capa sigmoide interna, lo que es más estable numéricamente para problemas de clasificación binaria, especialmente con datasets desbalanceados como el nuestro
        )

    def forward(self, video_x, audio_x, text_x):
        """
        Flujo hacia delante de los datos a través de la red.
        """
        # 1. Proyectamos cada modalidad a `proj_dim` dimensiones
        v_emb = self.visual_adapter(video_x) 
        a_emb = self.audio_adapter(audio_x)
        t_emb = self.text_adapter(text_x)
        
        # 2. Concatenación ----- EARLY FUSION -----
        ####### REGLA DE torch.cat: TODAS LAS DIMENSIONES QUE NO SE ESTÁN CONCATENANDO DEBEN SER IDÉNTICAS########
        # Por tanto, el batch_size (dim=0) debe ser igual para las tres modalidades. La única dimensión que se concatena es la de las características (dim=1)
        fused_feat = torch.cat((v_emb, a_emb, t_emb), 
                               dim=1# con dim=1 significa que "pegamos" las características (columnas), manteniendo el Batch (filas) intacto
                            )
        
        # 3. PASAMOS A LA CLASIFICACIÓN FINAL (MLP):
        output = self.mlp(fused_feat)
        
        return output
    
## NOTA FINAL: Gracias a la definición de PyTorch, heredando de nn.Module para definir la clase,
## el MLP se entrena de manera conjunta con los adaptadores, es decir, el gradiente se propaga a través de toda la red (adaptadores + MLP) durante el proceso de entrenamiento, lo que permite que tanto los adaptadores como el MLP aprendan a optimizarse conjuntamente para la tarea de clasificación.


# ------------ SEGUNDA ESTRATEGIA DE FUSIÓN: LATE FUSION ----------
# En esta estrategia, cada modalidad se procesa de forma independiente hasta el final, y luego se combinan las decisiones (logits) de cada modalidad
class LateFusionBase(nn.Module):
    """
    Arquitectura de Fusión Tardía (Late Fusion).
    Procesa cada modalidad de forma independiente hasta el final.
    - Instancia los adaptadores para proyectar a `proj_dim` dimensiones.
    - Pasa cada representación unimodal por su propio clasificador (MLP unimodal).
    - Promedia los logits finales antes de aplicar la función de pérdida.
    """
    def __init__(self, visual_dim=2048, audio_dim=768, text_dim=768, proj_dim=512, hidden_lstm=512, hidden_mlp=128, dropout_prob=0.5, lstm_layers=1):
        super(LateFusionBase, self).__init__()

        # 1. Instanciamos los Adaptadores (igual que en Early Fusion anterior)
        self.visual_adapter = VisualAdapter(input_dim=visual_dim, projection_dim=proj_dim, hidden_lstm=hidden_lstm, dropout_prob=dropout_prob, lstm_layers=lstm_layers)
        self.audio_adapter = AudioAdapter(input_dim=audio_dim, projection_dim=proj_dim, hidden_lstm=hidden_lstm, dropout_prob=dropout_prob, lstm_layers=lstm_layers)
        self.text_adapter = TextAdapter(input_dim=text_dim, projection_dim=proj_dim, dropout_prob=dropout_prob)
        
        # 2. Clasificadores Independientes (Unimodales)
        # En lugar de un MLP de `proj_dim`*3 dimensiones de entrada, creamos tres pequeños por cada modalidad.
        # Cada uno toma las características de su modalidad y devuelve 1 valor (logit):

        ## Arquitectura MLP VISUAL, AUDIO, TEXTUAL:
        ## - Capa 1 LINEAL: Reducción de `proj_dim` a `hidden_mlp` dimensiones. 
        ##     --> ReLU: Para introducir no linealidad.
        ##     --> Dropout: Para prevenir el sobreajuste, dado el tamaño pequeño del dataset.
        ## - Capa 2 LINEAL: Reducción de `hidden_mlp` dimensiones a 1 dimensión (logit)
        ##    ---> SALIDA: Logit (valor real)
        
        self.visual_clf = nn.Sequential(
            nn.Linear(proj_dim, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_mlp, 1) # Salida logit visual
        )
        
        self.audio_clf = nn.Sequential(
            nn.Linear(proj_dim, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_mlp, 1) # Salida logit acústica
        )
        
        self.text_clf = nn.Sequential(
            nn.Linear(proj_dim, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_mlp, 1) # Salida logit textual
        )

    def forward(self, video_x, audio_x, text_x):
        """
        Flujo hacia delante de los datos a través de la red.
        """
        # 1. Extracción de características 
        v_emb = self.visual_adapter(video_x) 
        a_emb = self.audio_adapter(audio_x)
        t_emb = self.text_adapter(text_x)
        
        # 2. Clasificación independiente ----- LATE FUSION -----
        v_logit = self.visual_clf(v_emb)
        a_logit = self.audio_clf(a_emb)
        t_logit = self.text_clf(t_emb)
        
        # 3. Fusión de decisiones (Promedio de los logits)
        # Al promediar los logits matemáticamente puros (antes de la sigmoide),
        # mantenemos la compatibilidad con BCEWithLogitsLoss, que es la función de pérdida que hemos aplicado en de train.py
        final_logit = (v_logit + a_logit + t_logit) / 3.0
        
        return final_logit


# ------------ TERCERA ESTRATEGIA DE FUSIÓN: ATTENTION FUSION ----------

class AttentionFusionBase(nn.Module):
    """
    Arquitectura de Fusión mediante Atención (Additive/Soft-Attention or Bahdanau Attention).
    Esta red aprende a ponderar dinámicamente la importancia de cada modalidad (Vídeo, Audio, Texto)
    para cada muestra de forma independiente
    """
    def __init__(self, visual_dim=2048, audio_dim=768, text_dim=768, proj_dim=512, hidden_lstm=512, hidden_mlp=128, dropout_prob=0.5, lstm_layers=1):
        super(AttentionFusionBase, self).__init__()

        # 1. Instanciamos los Adaptadores (lo mismo que hemos hecho para cada una de las estrategias, la salida es de `proj_dim` dims por cada modalidad)
        self.visual_adapter = VisualAdapter(input_dim=visual_dim, projection_dim=proj_dim, hidden_lstm=hidden_lstm, dropout_prob=dropout_prob, lstm_layers=lstm_layers)
        self.audio_adapter = AudioAdapter(input_dim=audio_dim, projection_dim=proj_dim, hidden_lstm=hidden_lstm, dropout_prob=dropout_prob, lstm_layers=lstm_layers)
        self.text_adapter = TextAdapter(input_dim=text_dim, projection_dim=proj_dim, dropout_prob=dropout_prob)
        
        # 2. Red de Atención  y Clasificador Final: 

        ## Arquitectura de ATENCIÓN + MLP:
        ##
        ##              ---------------- MECANISMO DE ATENCIÓN ADITIVO ----------------
        ##
        ## - Capa 1 LINEAL: CAPA DE ATENCIÓN. Reducción de `proj_dim` a `hidden_mlp` dimensiones, por cada modalidad por separado. Reducción a las `hidden_mlp` características más relevantes.
        ##      --> Tanh(): Función de activación Tangente Hiperbólica. Matemáticamente, coge el vector de salida de la anterior capa lineal y proyecta los valores entre -1 y 1. Esto permite que haya valores positivos y negativos, ya que los negativos permiten que la red aprenda los "pesos" de cada modalidad de forma estable y con mayor contraste entre aquello relevante (más cercano a 1) y ruido (más cercano a -1).
        ## - Capa 2 LINEAL: Score de Atención: Recibe de entrada un vector de `hidden_mlp` dimensiones filtradas por Tanh() -> De salida obtenemos un escalar (raw score o energía), que es la puntuación que le da la red de atención a dicha modalidad concreta.
        ##
        ##                          -------------- CLASIFICADOR MLP ----------------
        ## - Capa 1 LINEAL: Reducción del vector fusionado a `hidden_mlp` dims. 
        ##     --> ReLU: Para introducir no linealidad.
        ##     --> Dropout: Para prevenir el sobreajuste.
        ## - Capa 2 LINEAL: Reducción de `hidden_mlp` dimensiones a 1--> LOGIT FINAL DE LA RED!!!


        ##              ---------------- MECANISMO DE ATENCIÓN ADITIVO ----------------

        # Recibe una modalidad de `proj_dim` dims y devuelve 1 único valor (el score sin procesar, logit)
        ## ENTRADA: (Batch, 3, `proj_dim`) --> SALIDA: (Batch, 3, 1)
        self.attention_layer = nn.Sequential(
            nn.Linear(proj_dim, hidden_mlp),
            nn.Tanh(),
            nn.Linear(hidden_mlp, 1)
        )

        ##                          -------------- CLASIFICADOR MLP ----------------
        # Recibe el vector fusionado final (que sigue siendo de `proj_dim` dims) y obtiene la predicción final
        ## ENTRADA: (Batch, `proj_dim`) --> (Batch, `hidden_mlp`) --> (Batch, 1)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_mlp, 1) # Salida logit final
        )

        
    def forward(self, video_x, audio_x, text_x):
        """
        Flujo hacia delante de los datos a través de la red.
        """
        # 1. Extracción de características con adapters ---> (Batch, `proj_dim`)
        v_emb = self.visual_adapter(video_x) 
        a_emb = self.audio_adapter(audio_x)
        t_emb = self.text_adapter(text_x)
        
        # 2. Apilamos las modalidades en una nueva dimensión:
        # Apilamos esos 3 tensores de cada modalidad a un único "bloque" de tamaño: (Batch, 3 modalidades, `proj_dim` características)
        stacked_embs = torch.stack([v_emb, a_emb, t_emb], dim=1)
        
        # 3. Calculamos los "Scores" de Atención
        # Le pasamos el bloque a la red de atención. Nos devuelve un score por modalidad: (Batch, 3, 1)
        attn_scores = self.attention_layer(stacked_embs)
        
        # 4. Aplicamos Softmax: 
        # Con softmax se mapean esos scores para que los 3 valores sumen exactamente 1.0 (interpretándolos así como probabilidades, por ej.: 0.7, 0.2, 0.1, por cada modalidad ) 
        attn_weights = torch.softmax(attn_scores, dim=1) # dim=1 indica que calcule los valores de las 3 modalidades
        
        # 5. Fusión ponderada
        # Multiplicamos cada modalidad por su probabilidad de atención (0.7 * v_emb, etc...)
        weighted_embs = stacked_embs * attn_weights
        
        # Sumamos las tres modalidades ya ponderadas para aplastar el bloque de nuevo a (Batch, `proj_dim`)
        # Se suma ya que estamos calculando una media ponderada:
        context_vector = torch.sum(weighted_embs, dim=1)
        
        # 6. Clasificación Final
        # Pasamos el vector de contexto final por el clasificador
        final_logit = self.classifier(context_vector)
        
        return final_logit