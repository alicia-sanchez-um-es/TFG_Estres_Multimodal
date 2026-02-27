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
    - Instanciar los adaptadores para cada modalidad (visual, audio, texto) que proyectan las características a un espacio común de 512 dimensiones.
    - Concatenar las características proyectadas de las tres modalidades (512 + 512 + 512 = 1536 dimensiones).
    - Pasar la representación multimodal fusionada a través de un MLP para la clasificación final.

    Devuelve:
    - Un valor entre 0 y 1 que representa la probabilidad de que el vídeo esté etiquetado como Estrés (1) o No Estrés (0).
    """
    def __init__(self, visual_dim=2048, audio_dim=768, text_dim=768, proj_dim=512):
        super(EarlyFusionBase, self).__init__()

        # Este constructor sirve de "pegamento" para los adaptadores creados en adapters.py y el MLP FINAL para clasificación definido a continuación.
        
        # 1. Instanciamos los Adaptadores 
        self.visual_adapter = VisualAdapter(input_dim=visual_dim, projection_dim=proj_dim)
        self.audio_adapter = AudioAdapter(input_dim=audio_dim, projection_dim=proj_dim)
        self.text_adapter = TextAdapter(input_dim=text_dim, projection_dim=proj_dim)
        
        # 2. El Clasificador Final (MLP)
        # La entrada es la suma de los tres espacios proyectados: 512 + 512 + 512 = 1536
        fusion_dim = proj_dim * 3
        
        ## Arquitectura del MLP:
        ## - Capa 1: Reducción de 1536 a 512 dimensiones. Primer paso hacia una reducción gradual de la dimensionalidad, permite que la red aprenda a condensar la información multimodal.
        ##     --> ReLU: Para introducir no linealidad, clave para que la red aprenda relaciones no lineales entre las características multimodales.
        ##     --> Dropout: Del 50%. Para prevenir el sobreajuste, dado el tamaño relativamente pequeño del dataset, debemos incluir regularización.
        ## - Capa 2: Reducción de 512 a 128 dimensiones. Se continúa la reducción gradual.
        ##     --> ReLU: Para mantener la capacidad de modelar relaciones complejas.
        ##     --> Dropout: Del 50% para seguir previniendo el sobreajuste.
        ## - Capa Final: Reducción de 128 a 1 dimensión. Esta neurona final devuelve un valor bruto (logit).
        ##     --> Sigmoid: Para convertir el logit en una probabilidad entre 0 y 1, que es lo que necesitamos para la clasificación binaria de estrés/no estrés

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
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
        # 1. Proyectamos cada modalidad a 512 dimensiones
        v_emb = self.visual_adapter(video_x) 
        a_emb = self.audio_adapter(audio_x)
        t_emb = self.text_adapter(text_x)
        
        # 2. Concatenación ----- EARLY FUSION -----
        ####### REGLA DE torch.cat: TODAS LAS DIMENSIONES QUE NO SE ESTÁN CONCATENANDO DEBEN SER IDÉNTICAS########
        # Por tanto, el batch_size (dim=0) debe ser igual para las tres modalidades. La única dimensión que se concatena es la de las características (dim=1), que pasa de 512 a 1536 al juntar las tres modalidades.
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
    - Instancia los adaptadores para proyectar a 512 dimensiones.
    - Pasa cada representación unimodal por su propio clasificador (MLP unimodal).
    - Promedia los logits finales antes de aplicar la función de pérdida.
    """
    def __init__(self, visual_dim=2048, audio_dim=768, text_dim=768, proj_dim=512):
        super(LateFusionBase, self).__init__()

        # 1. Instanciamos los Adaptadores (igual que en Early Fusion anterior)
        self.visual_adapter = VisualAdapter(input_dim=visual_dim, projection_dim=proj_dim)
        self.audio_adapter = AudioAdapter(input_dim=audio_dim, projection_dim=proj_dim)
        self.text_adapter = TextAdapter(input_dim=text_dim, projection_dim=proj_dim)
        
        # 2. Clasificadores Independientes (Unimodales)
        # En lugar de un MLP de 1536 dimensiones de entrada, creamos tres pequeños por cada modalidad.
        # Cada uno toma las 512 características de su modalidad y devuelve 1 valor (logit):
        
        self.visual_clf = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1) # Salida logit visual
        )
        
        self.audio_clf = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1) # Salida logit acústica
        )
        
        self.text_clf = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1) # Salida logit textual
        )

    def forward(self, video_x, audio_x, text_x):
        # 1. Extracción de características (512 dims cada una)
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