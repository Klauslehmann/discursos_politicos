
#del pre_process_text

import sys
sys.path.append('scripts/')
import time
from gensim.models.fasttext import load_facebook_model
import string
from helpers import pre_process_text
from helpers import get_centroid
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import pickle
import multiprocessing


# Cargar modelo de embeddings
wordvectors = load_facebook_model('data/embeddings-s-model.bin') 


# Cargar datos editados
df =  pd.read_feather("/mnt/c/proyectos_personales/discursos_politicos/data/clean_data/edicion_inicial_light.feather")
size = df.shape
print(size)

# Pre procesar texto 
cpus = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cpus)
tokenized_text = pool.map(pre_process_text,df.texto_dep[0:1000])

#tokenized_text =  [pre_process_text(text)  for text in df.texto_dep[0:1000]]

tokenized = list(map(lambda x:x[0], tokenized_text)) 
original_text = list(map(lambda x:x[1], tokenized_text))
original_text[1]

# Convertir textos procesados en embeddings y luego buscar el centroide
text_vectors = [[wordvectors.wv[word] for word in sentence ]  for sentence in tokenized]
sentences_centroids = [[get_centroid(sentence) for sentence in text ] for text  in text_vectors]


# Guardar centroides 
with open("data/centroids", "wb") as fp:
  pickle.dump(sentences_centroids, fp)

# Guardar tokenización
with open("data/tokenization", "wb") as fp:
  pickle.dump(tokenized, fp)

# Guardar textos originales separados en oraciones
with open("data/original_sentences", "wb") as fp:
  pickle.dump(original_text, fp)


