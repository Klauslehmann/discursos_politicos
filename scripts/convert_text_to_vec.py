
#del pre_process_text

import sys
sys.path.append('scripts/')
import time
from gensim.models.fasttext import load_facebook_model
import string
from helpers import pre_process_text
from helpers import parallel_text_processing
from helpers import get_centroid
from helpers import convert_to_vec
from helpers import flatten
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import pickle
import multiprocessing
from functools import partial

# Argumentos del script
m =  sys.argv[1]
m = "word"

# Cargar datos editados
df =  pd.read_feather("/home/klaus/discursos_politicos/data/edicion_inicial_light.feather")

# Pre procesar texto 
tokenized_text = parallel_text_processing(df.texto_dep, batch_size = 5000,  mode = m)
tokenized = list(map(lambda x:x[0], tokenized_text)) 
original_text = list(map(lambda x:x[1], tokenized_text))

# Cargar modelo de embeddings
wordvectors = load_facebook_model('/home/klaus/discursos_politicos/data/embeddings-m-model.bin') 

# Convertir textos procesados en embeddings y luego buscar el centroide
sentences_centroids_text = convert_to_vec(wordvectors, tokenized, mode = "text")  
sentences_centroids_sentence = convert_to_vec(wordvectors, tokenized, mode = "sentence")  

# Guardar centroides 
with open("/home/klaus/discursos_politicos/data/centroids_text", "wb") as fp:
  pickle.dump(sentences_centroids_text, fp)

with open("/home/klaus/discursos_politicos/data/centroids_sentence", "wb") as fp:
  pickle.dump(sentences_centroids_sentence, fp)

# Guardar tokenización
with open("/home/klaus/discursos_politicos/data/tokenization", "wb") as fp:
  pickle.dump(tokenized, fp)

# Guardar textos originales separados en oraciones
with open("/home/klaus/discursos_politicos/data/original_sentences", "wb") as fp:
  pickle.dump(original_text, fp)

del sentences_centroids_text
del sentences_centroids_sentence
del wordvectors 
del tokenized
del original_text
del tokenized_text
del df
