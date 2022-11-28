import pandas as pd
import sys
sys.path.append('scripts/')
from transformers import pipeline
import time


###############
# CREAR TÓPICOS 
###############

#df_full = pd.read_feather("data/score_full.feather")
#df_full .to_csv("data/score_full.csv")

df_full = pd.read_csv("data/score_full.csv")
df_filtrado = df_full[df_full["n_words"] >= 3 ].reset_index()

# Cargar el modelo para encontrar tópicos
classifier = pipeline("zero-shot-classification", 
                       model="Recognai/zeroshot_selectra_medium")


# Defunir tópicos 
topics = ["salud", "educación", "deporte", 
         "medioambiente", "impuestos", 
         "cultura", "pensiones", 
         "sindicalismo", "transporte", "familia", "aborto"]

# Seleccionar una muestra para crear los tópicos
n = df_filtrado.shape[0]
n = 10
tx  = df_filtrado.text.sample(n = n, random_state = 1,)

# Clasificar cada texto con un tópico
tic = time.time()
topic = [classifier(text, candidate_labels = topics ) 
         for text in tx]
toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))


# Rescatar la etiqueta y el puntaje asignado por el modelo a la etiqueta
labels = [t["labels"][0] for t in topic]
score_topic = [t["scores"][0] for t in topic]
indices = tx.index

# Constuir dataframe con un subconjunto de textos clasificados con tópico 
dic = {"id_phrase": df_filtrado.id_phrase[indices],
       "id_original_text": df_filtrado.id_speech[indices],
       "fecha": df_filtrado.fecha[indices],
       "name": df_filtrado.nombre[indices],
       "text": tx, 
       "topic": labels,
       "score_topic": score_topic,
       "score":df_filtrado.score[indices],
       "pole": df_filtrado.polo[indices],
       "emotion": df_filtrado.cos_affect[indices], 
       "cognition": df_filtrado.cos_cognitive[indices],
       "age": df_filtrado.edad_actual[indices],
       "year": df_filtrado.anio[indices],
       "sex": df_filtrado.sexo[indices] 
       }
 
df_topics= pd.DataFrame(dic ).reset_index()

#################
# GUARDAR DATOS #
#################
df_filtrado .to_feather("data/score_filtered.feather")
df_filtrado .to_csv("data/score_filtered.csv")

