import torch
import torch.nn as nn
# Importamos los adaptadores creados desde adapters.py:
from .adapters import VisualAdapter, AudioAdapter, TextAdapter

# ------------ PRIMERA ESTRATEGIA DE FUSIÓN: EARLY FUSION ----------
# En esta estrategia, se concatenan los embeddings proyectados a un espacio común antes de la clasificación

class EarlyFusion(nn.Module):
    """
    Arquitectura de Fusión Temprana (Early Fusion) para audio, vídeo y/o texto.
    Concatena las representaciones unimodales a nivel de características.
    Esta clase lo que hace es:
    - Instanciar los adaptadores para cada modalidad (visual/audio/texto) que proyectan las características a un espacio común.
    - Concatenar las características proyectadas de las dos o tres modalidades.
    - Pasar la representación multimodal fusionada a través de un MLP para la clasificación final.

    Devuelve:
    - Un logit en bruto.
    """
    def __init__(self, visual_dim=2048, audio_dim=768, text_dim=768, proj_dim=512, hidden_mlp=128, dropout_prob=0.5, modalidades = ['video','audio','texto']):
        super(EarlyFusion, self).__init__()
        
        self.modalidades = modalidades

        # 1. Instanciamos los Adaptadores
        # Este constructor sirve de "puente" para los adaptadores creados en adapters.py y el MLP FINAL para clasificación definido a continuación, llevandi a cabo la concatenación 
        # Antes de definir los adaptadores, debemos calcular la dimensión de la capa oculta con un valor razonable derivado de las dimensiones iniciales del vector de características recibido por la LSTM:

        if 'audio' in self.modalidades :
            hidden_lstm_audio = max(proj_dim, audio_dim // 4) 
            self.audio_adapter = AudioAdapter(input_dim=audio_dim, projection_dim=proj_dim, hidden_lstm=hidden_lstm_audio, dropout_prob=dropout_prob)
        
        if 'video' in self.modalidades :
            hidden_lstm_visual = max(proj_dim, visual_dim // 4) # Para embeddings iniciales de alta dimensionalidad (por ejemplo, ResNet con 2048 dimensiones o EfficientNet con 1280 dimensiones) la LSTM actúa como paso intermedio para no comprimir de golpe a proj_dim, sino de manera gradual
            self.visual_adapter = VisualAdapter(input_dim=visual_dim, projection_dim=proj_dim, hidden_lstm=hidden_lstm_visual, dropout_prob=dropout_prob)
         
        if 'texto' in self.modalidades:
            self.text_adapter = TextAdapter(input_dim=text_dim, projection_dim=proj_dim, dropout_prob=dropout_prob)
        
        # 2. El Clasificador Final (MLP)
        # La entrada es la concatenación de los dos/tres espacios proyectados
        fusion_dim = proj_dim * len(self.modalidades)

        intermediate_dim = proj_dim
        
        ## Arquitectura del MLP:
        ## - Capa 1 LINEAL: Reducción de proj_dim*3/proj_dim*2 a `intermediate_dim` (proj_dim) dimensiones. Primer paso hacia una reducción gradual de la dimensionalidad, permite que la red aprenda a condensar la información multimodal
        ##     --> BatchNorm1d: Normalización por lotes
        ##     --> ReLU: Para introducir no linealidad, clave para que la red aprenda relaciones no lineales entre las características multimodales
        ##     --> Dropout: Para prevenir el sobreajuste, dado el tamaño relativamente pequeño del dataset, debemos incluir regularización
        ## - Capa 2 LINEAL: Reducción de `intermediate_dim` a `hidden_mlp`. Se continúa la reducción gradual
        ##     --> BatchNorm1d: Normalización por lotes
        ##     --> ReLU: Para mantener la capacidad de modelar relaciones complejas
        ##     --> Dropout: Para seguir previniendo el sobreajuste
        ## - Capa Final LINEAL: Reducción de `hidden_mlp` a 1 dimensión. Esta neurona final devuelve un valor bruto (logit).

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(intermediate_dim, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            # Capa final: 1 neurona que devuelve un valor bruto (logit)
            nn.Linear(hidden_mlp, 1)
            
            # POC inicial, se probó con una capa final sigmoide como prueba inicial, obteniendo un rendimiento muy bajo para nuestro dataset desbalanceado
            # Sigmoide para convertir ese valor en probabilidad de estrés (0 a 1)
            # nn.Sigmoid() 

            # CAMBIO A LOGITS: Se decidió dejar la capa final sin activación (logits) para utilizar BCEWithLogitsLoss, que combina la función de pérdida de entropía cruzada con una capa sigmoide interna, lo que es más estable numéricamente para problemas de clasificación binaria, especialmente con datasets desbalanceados como el nuestro
        )

    def forward(self, video_x = None, audio_x = None, text_x = None):
        """
        Flujo hacia delante de los datos a través de la red.
        """
        embs = []
        # 1. Proyectamos cada modalidad a `proj_dim` dimensiones
        if 'video' in self.modalidades and video_x is not None:
            embs.append(self.visual_adapter(video_x))
        if 'audio' in self.modalidades and audio_x is not None:
            embs.append(self.audio_adapter(audio_x))
        if 'texto' in self.modalidades and text_x is not None:
            embs.append(self.text_adapter(text_x))
        
        # 2. Concatenación ----- EARLY FUSION -----
        ####### REGLA DE torch.cat: TODAS LAS DIMENSIONES QUE NO SE ESTÁN CONCATENANDO DEBEN SER IDÉNTICAS########
        # Por tanto, el batch_size (dim=0) debe ser igual para las tres/dos modalidades. La única dimensión que se concatena es la de las características (dim=1)
        vector_fusionado = torch.cat(embs, 
                               dim=1# con dim=1 significa que "pegamos" las características (columnas), manteniendo el Batch (filas) intacto
                            )
        
        # 3. PASAMOS A LA CLASIFICACIÓN FINAL (MLP):
        output = self.mlp(vector_fusionado)
        
        return output
    
## NOTA FINAL: Gracias a la definición de PyTorch, heredando de nn.Module para definir la clase,
## el MLP se entrena de manera conjunta con los adaptadores, es decir, el gradiente se propaga a través de toda la red (adaptadores + MLP) durante el proceso de entrenamiento, lo que permite que tanto los adaptadores como el MLP aprendan a optimizarse conjuntamente para la tarea de clasificación


# ------------ SEGUNDA ESTRATEGIA DE FUSIÓN: LATE FUSION ----------
# En esta estrategia, cada modalidad se procesa de forma independiente hasta el final, y luego se combinan las decisiones (logits) de cada modalidad
class LateFusion(nn.Module):
    """
    Arquitectura de Fusión Tardía (Late Fusion).
    Procesa cada modalidad de forma independiente hasta el final.
    - Instancia los adaptadores para proyectar a `proj_dim` dimensiones.
    - Pasa cada representación unimodal por su propio clasificador (MLP unimodal).
    - Aplica 3 técnicas distintas sobre los logits finales.
    """
    def __init__(self, visual_dim=2048, audio_dim=768, text_dim=768, proj_dim=512, hidden_mlp=128, dropout_prob=0.5, fusion_mode = 'promedio', modalidades = ['video','audio','texto']):
        super(LateFusion, self).__init__()

        self.fusion_mode = fusion_mode
        self.modalidades = modalidades
        num_mods = len(modalidades) # 2 o 3

        # 1. Instanciamos los Adaptadores (igual que en Early)

        # 2. Clasificadores Independientes (Unimodales)
        # En lugar de un MLP de `proj_dim`*3 /`proj_dim`*2 dimensiones de entrada, creamos tres pequeños por cada modalidad.
        # Cada uno toma las características de su modalidad y devuelve 1 valor (logit):

        ## Arquitectura MLP VISUAL, AUDIO, TEXTUAL:
        ## - Capa 1 LINEAL: Reducción de `proj_dim` a `hidden_mlp` dimensiones. 
        ##     --> ReLU: Para introducir no linealidad.
        ##     --> Dropout: Para prevenir el sobreajuste, dado el tamaño pequeño del dataset.
        ## - Capa 2 LINEAL: Reducción de `hidden_mlp` dimensiones a 1 dimensión (logit)
        ##    ---> SALIDA: Logit (valor real)
        

        if 'audio' in modalidades :
            hidden_lstm_audio = max(proj_dim, audio_dim // 4)
            self.audio_adapter = AudioAdapter(input_dim=audio_dim, projection_dim=proj_dim, hidden_lstm=hidden_lstm_audio, dropout_prob=dropout_prob)
            self.audio_clf = nn.Sequential(
                nn.Linear(proj_dim, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_mlp, 1) # Salida logit acústica
            )
        
        if 'video' in modalidades :
            hidden_lstm_visual = max(proj_dim, visual_dim // 4) # Para embeddings iniciales de alta dimensionalidad (por ejemplo, ResNet con 2048 dimensiones o EfficientNet con 1280 dimensiones) la LSTM actúa como paso intermedio para no comprimir de golpe a proj_dim, sino de manera gradual
            self.visual_adapter = VisualAdapter(input_dim=visual_dim, projection_dim=proj_dim, hidden_lstm=hidden_lstm_visual, dropout_prob=dropout_prob)
            self.visual_clf = nn.Sequential(
                nn.Linear(proj_dim, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_mlp, 1) # Salida logit visual
            )
         
        if 'texto' in modalidades:
            self.text_adapter = TextAdapter(input_dim=text_dim, projection_dim=proj_dim, dropout_prob=dropout_prob)
            self.text_clf = nn.Sequential(
                nn.Linear(proj_dim, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_mlp, 1) # Salida logit textual
            )
        

        # Añadimos una capa "extra" de fusión para Regresión Logística:
        if self.fusion_mode == 'logistica' :
            # Recibe los 3 o 2 logits (v_logit/a_logit/t_logit) y devuelve 1 logit final
            # Con esto, la red aprenderá coeficientes de importancia 
            self.logistic_fusion = nn.Linear(num_mods, 1)

    def forward(self, video_x=None, audio_x=None, text_x=None, return_latent = False):
        """
        Flujo hacia delante de los datos a través de la red.
        """
        logits = []
        # Extracción de logits INDEPENDIENTES (LATE FUSION)
        if 'video' in self.modalidades and video_x is not None:
            v_emb = self.visual_adapter(video_x)
            logits.append(self.visual_clf(v_emb))
        if 'audio' in self.modalidades and audio_x is not None:
            a_emb = self.audio_adapter(audio_x)
            logits.append(self.audio_clf(a_emb))
        if 'texto' in self.modalidades and text_x is not None:
            t_emb = self.text_adapter(text_x)
            logits.append(self.text_clf(t_emb))

        # Juntamos todos los logits en un solo tensor de forma (Batch, num_modalidades):
        stacked_logits = torch.cat(logits, dim = 1)


        ############################# TÉCNICA 1: Fusión mediante voto mayoritario #####################################

        if self.fusion_mode == 'voto' :

            # Con esta técnica, devolvemos el logit como la confianza media
            # de aquellos que votaron a la clase ganadora. Si gana ESTRÉS, 
            # se devuelve un logit positivo con la confianza media obtenida 
            # de los que votaron ESTRÉS. Si gana NO ESTRÉS, devolvemos un logit negativo (*-1)
            # con la confianza media de los que votaron NO ESTRÉS. 

            # PASO 1: Convertimos los logits a probabilidades con Sigmoide:
            probs = torch.sigmoid(stacked_logits)

            # PASO 2: Obtenemos las predicciones con umbral 0.5:
            votos = (probs >= 0.5).float()  # 1 = estrés, 0 = no estrés
            

            # PASO 3: Voto mayoritario (sumamos todas las predicciones):
            total_votos = torch.sum(votos, dim=1, keepdim=True) # Podemos obtener 0 (TODOS COINCIDEN EN NO ESTRÉS), 1 (al menos 1 estrés), 2 (2 dicen estrés), 3 (TODOS COINCIDEN EN ESTRÉS, en el caso trimodal)

            # PASO 4: Aplicamos máscara:
            # Para 2 modalidades (bimodal):
            # MAYORÍA es 2 (en caso de empate 1-1 gana la clase 0 por defecto)
            # Para 3 modalidades:
            # MAYORÍA es 2 (nunca van a empatar)
            ## A FAVOR -> ESTRÉS (logit positivo), EN CONTRA -> NO ESTRÉS (logit negativo)
            umbral = len(self.modalidades) // 2 + 1 
            estres_mask = (total_votos >= umbral)  # (Batch, 1) bool, si es TRUE es ESTRÉS, si es FALSE es NO ESTRÉS

            # PASO 5: Desempate por confianza: ¿cómo de seguros estaban los que votaron con la mayoría?
            # Reconstruimos el logit final para BCEWithLogitsLoss ponderado por la confianza de la mayoría ganadora:
            # Confianza = probabilidad media de los que votaron igual que la mayoría
            conf_estres = torch.sum(probs * votos, dim=1, keepdim=True) / (total_votos + 1e-8)
            # Sumamos las probabilidades de los que votaron no estrés y dividimos por su cantidad
            conf_no_estres = torch.sum((1 - probs) * (1 - votos), dim=1, keepdim=True) / (len(self.modalidades) - total_votos + 1e-8)
            ### LOGIT: positivo si mayoría vota estrés, negativo si no 
            final_logit = torch.where(estres_mask, conf_estres, -conf_no_estres) # Si estres_mask es TRUE es porque la mayoría votaron ESTRÉS, por tanto se devuelve logit positivo conf_estres que es la confianza media de los que votaron estrés. Pero si estres_mask es False, es porque la mayoría votaron NO ESTRÉS, por tanto devolvemos logit negativo con la confianza media de los que votaron no estrés -conf_no_estres

        ############################ TÉCNICA 2: Fusión mediante media aritmética de los logits ###################################
        
        elif self.fusion_mode == 'promedio' :
        # Fusión de decisiones (Promedio de los logits)
        # Al promediar los logits matemáticamente puros (antes de la sigmoide), mantenemos la compatibilidad con BCEWithLogitsLoss
            final_logit = torch.mean(stacked_logits, dim=1, keepdim=True)

        ############################# TÉCNICA 3: Fusión ponderada aprendida (Regresión Logística) #####################################
            # Matemáticamente, una regresión logística clásica es la suma ponderada de las entradas 
            # seguida de una sigmoide. Al tener la función de pérdida BCEWithLogitsLoss, esta cuenta
            # con la sigmoide internamente, por tanto, aplicando aquí una suma ponderada aprendida a través de una capa lineal,
            # conseguimos técnicamente lo mismo que una regresión logística

        elif self.fusion_mode == 'logistica':
            final_logit = self.logistic_fusion(stacked_logits)

        if return_latent:
            return final_logit, stacked_logits

        return final_logit


# ------------ TERCERA ESTRATEGIA DE FUSIÓN: ATTENTION FUSION ----------

class AttentionFusion(nn.Module):
    """
    Arquitectura de Fusión mediante Atención Aditiva (de Bahdanau) entre modalidades.
    Esta red aprende a ponderar dinámicamente la importancia de cada modalidad (Vídeo, Audio, Texto)
    para cada muestra de forma independiente inspirándose en el mecanismo de atención aditiva de Bahdanau.
    """
    def __init__(self, visual_dim=2048, audio_dim=768, text_dim=768, proj_dim=512, hidden_mlp=128, dropout_prob=0.5, modalidades = ['video','audio','texto']):
        super(AttentionFusion, self).__init__()

        self.modalidades = modalidades

        # 1. Instanciamos los Adaptadores:
        if 'audio' in self.modalidades :
            hidden_lstm_audio = max(proj_dim, audio_dim // 4)
            self.audio_adapter = AudioAdapter(input_dim=audio_dim, projection_dim=proj_dim, hidden_lstm=hidden_lstm_audio, dropout_prob=dropout_prob)
        
        if 'video' in self.modalidades :
            hidden_lstm_visual = max(proj_dim, visual_dim // 4) # Para embeddings iniciales de alta dimensionalidad (por ejemplo, ResNet con 2048 dimensiones o EfficientNet con 1280 dimensiones) la LSTM actúa como paso intermedio para no comprimir de golpe a proj_dim, sino de manera gradual
            self.visual_adapter = VisualAdapter(input_dim=visual_dim, projection_dim=proj_dim, hidden_lstm=hidden_lstm_visual, dropout_prob=dropout_prob)
         
        if 'texto' in self.modalidades:
            self.text_adapter = TextAdapter(input_dim=text_dim, projection_dim=proj_dim, dropout_prob=dropout_prob)

        # 2. Red de Atención  y Clasificador Final: 

        ## Arquitectura de ATENCIÓN + MLP:
        ##
        ##              ---------------- MECANISMO DE ATENCIÓN ADITIVO ----------------
        ##
        ## - Capa 1 LINEAL: CAPA DE ATENCIÓN. Reducción de `proj_dim` a `hidden_mlp` dimensiones, por cada modalidad por separado. Reducción a las `hidden_mlp` características más relevantes
        ##      --> Tanh(): Función de activación Tangente Hiperbólica. Matemáticamente, coge el vector de salida de la anterior capa lineal y proyecta los valores entre -1 y 1. Esto permite que haya valores positivos y negativos, ya que los negativos permiten que la red aprenda los "pesos" de cada modalidad de forma estable y con mayor contraste entre aquello relevante (más cercano a 1) y ruido (más cercano a -1)
        ## - Capa 2 LINEAL: Score de Atención: Recibe de entrada un vector de `hidden_mlp` dimensiones filtradas por Tanh() -> De salida obtenemos un escalar (raw score o energía), que es la puntuación que le da la red de atención a dicha modalidad concreta
        ##
        ##                          -------------- CLASIFICADOR MLP ----------------
        ## - Capa 1 LINEAL: Reducción del vector fusionado a `hidden_mlp` dims
        ##     --> ReLU: Para introducir no linealidad
        ##     --> Dropout: Para prevenir el sobreajuste
        ## - Capa 2 LINEAL: Reducción de `hidden_mlp` dimensiones a 1--> LOGIT FINAL DE LA RED!!!


        ##              ---------------- MECANISMO DE ATENCIÓN ADITIVO ----------------

        # Recibe una modalidad de `proj_dim` dims y devuelve 1 único valor (el score sin procesar, logit)
        ## ENTRADA: (Batch, num_mods, `proj_dim`) --> SALIDA: (Batch, num_mods, 1)
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

        
    def forward(self, video_x=None, audio_x=None, text_x=None, return_attention=False):
        """
        Flujo hacia delante de los datos a través de la red.
        """
        # 1. Extracción de características con adapters ---> (Batch, `proj_dim`)
        embs = []
        if 'video' in self.modalidades and video_x is not None:
            embs.append(self.visual_adapter(video_x))
        if 'audio' in self.modalidades and audio_x is not None:
            embs.append(self.audio_adapter(audio_x))
        if 'texto' in self.modalidades and text_x is not None:
            embs.append(self.text_adapter(text_x))
        
        # 2. Apilamos las modalidades en una nueva dimensión:
        # Apilamos esos 3 tensores de cada modalidad a un único "bloque" de tamaño: (Batch, num modalidades, `proj_dim` características)
        stacked_embs = torch.stack(embs, dim=1)
        
        # 3. Calculamos los "Scores" de Atención
        # Le pasamos el bloque a la red de atención. Nos devuelve un score por modalidad: (Batch, 3, 1)
        attn_scores = self.attention_layer(stacked_embs) # PRODUCTO PUNTO A PUNTO
        
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

        # Para realizar el análisis de interpretabilidad, devolvemos los pesos de atención:
        if return_attention:
            return final_logit, attn_weights
        
        return final_logit