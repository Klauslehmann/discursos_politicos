
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
from functools import partial

# Argumentos del script
m =  sys.argv[1]
m = "word"

# Cargar modelo de embeddings
wordvectors = load_facebook_model('/home/klaus/discursos_politicos/data/embeddings-s-model.bin') 

# Cargar datos editados
df =  pd.read_feather("/home/klaus/discursos_politicos/data/edicion_inicial_light.feather")
df = df[0:50000]
size = df.shape
print(size)

# Pre procesar texto 
tokenized_text = parallel_text_processing(df.texto_dep)
tokenized = list(map(lambda x:x[0], tokenized_text)) 
original_text = list(map(lambda x:x[1], tokenized_text))
original_text[1]

# Convertir textos procesados en embeddings y luego buscar el centroide
text_vectors = [[wordvectors.wv[word] for word in sentence ]  for sentence in tokenized]
sentences_centroids = [[get_centroid(sentence) for sentence in text ] for text  in text_vectors]


# Guardar centroides 
with open("/home/klaus/discursos_politicos/data/centroids", "wb") as fp:
  pickle.dump(sentences_centroids, fp)

# Guardar tokenización
with open("/home/klaus/discursos_politicos/data/tokenization", "wb") as fp:
  pickle.dump(tokenized, fp)

# Guardar textos originales separados en oraciones
with open("/home/klaus/discursos_politicos/data/original_sentences", "wb") as fp:
  pickle.dump(original_text, fp)


