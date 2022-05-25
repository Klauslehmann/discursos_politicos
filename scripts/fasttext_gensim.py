
del pre_process_text

import sys
sys.path.append('scripts/')

from gensim.models.fasttext import load_facebook_model
import string
from helpers import pre_process_text
from helpers import get_centroid
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
# estoy haciendo una modificación desde colab
# Cargar datos editados
df =  pd.read_feather("/mnt/c/proyectos_personales/discursos_politicos/data/clean_data/edicion_inicial_light.feather")
size = df.shape
print(size)

# Crear tabla reducida, para hacer pruebas
reducida = df[0:500]

# Pre procesar texto 
tokenized =  [pre_process_text(text)  for text in reducida.texto_dep]
[len(t) for t in tokenized]


# Convertir textos procesados en embeddings y luego buscar el centroide
wordvectors = load_facebook_model('data/embeddings-s-model.bin') 
text_vectors = [[wordvectors.wv[word] for word in sentence ]  for sentence in tokenized]
sentences_centroids = [[get_centroid(sentence) for sentence in text ] for text  in text_vectors]

# Guardar centroides 
with open("data/centroids", "wb") as fp:
  pickle.dump(sentences_centroids, fp)



