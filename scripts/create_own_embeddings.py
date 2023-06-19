import sys
sys.path.append('scripts/')
from helpers import tokenizar
from helpers import flatten
import pandas as pd
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
#nltk.download('punkt')
import re
from tqdm import tqdm
import pickle
import time
import random


# Cargar archivo con todos los textos
df =  pd.read_feather("/home/klaus/discursos_politicos/data/edicion_inicial.feather")

# Sacar caracter molesto de salto de línea
text2 = [re.sub(r'\\n',' ', text)  for text in df["texto_dep"]]
text_sample = random.sample(text2, 150000)

del df
del text_sample 

# tokenizar divide un texto en oraciones y luego tokeniza las oraciones
tokenized_text =  [tokenizar(text) for text in tqdm(text2)]

# Se unen las listas, para adecuar el input a lo que necsita la función de fasttext
flat = flatten(tokenized_text)

del tokenized_text
# =============================================================================
# 
# # Guardar tokenización
# with open("/home/klaus/discursos_politicos/data/tokens_own_embeddings.pkl", 'wb') as f:
#     pickle.dump(flat, f)
# 
# # Cargar tokenización
# with open("/home/klaus/discursos_politicos/data/tokens_own_embeddings.pkl", 'rb') as f:
#     flat = pickle.load(f)
# 
# =============================================================================


# Este modelo entrena embeddings de 100 dimensiones, corriendo durante 10 épocas
start_time = time.time()

model = FastText(vector_size=100, window=3, min_count=2, workers = 15, min_n = 3, max_n=6)  # instantiate
model.build_vocab(corpus_iterable=flat)
model.train(corpus_iterable=flat, total_examples=len(flat ), epochs=20)

final_time = time.time()
print(final_time - start_time)


# Guardar el modelo de embeddings
fname = get_tmpfile("/home/klaus/discursos_politicos/data/embeddings-100-own.model")
model.save(fname)
model = FastText.load(fname)

print(model.wv['sal'])




