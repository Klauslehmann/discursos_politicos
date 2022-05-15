import sys
sys.path.append('scripts/')

from helpers import pre_process_text
import numpy as np
import spacy
nlp = spacy.load('es_core_news_md')

import pandas as pd
from nltk.corpus import stopwords

# Cargar datos editados
df =  pd.read_feather("/mnt/c/proyectos_personales/discursos_politicos/data/clean_data/edicion_inicial.feather")
size = df.shape
print(size)

# Crear tabla reducida, para hacer pruebas
reducida = df[0:10]

# Pre procesar texto. [AQUÍ SE PUEDE PROBAR CON CUALQUIER TEXTO.]
tokenized =  [pre_process_text(text)  for text in reducida.texto_dep ]




# Pasar las palabras por el stemmignde spacy
# cognitive_vectors_list = [nlp(word).vector for word in df_cognitive]
# cognitive_vectors_array = np.asarray(cognitive_vectors_list)
# cognitive_vector = np.mean(cognitive_vectors_array, axis=0)
# 
# 
# print(stopwords.words('spanish'))
