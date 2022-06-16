from numpy import dot
from numpy.linalg import norm
import pickle
from collections import Counter
import pandas as pd
import sys
sys.path.append('scripts/')
from helpers import get_cosine
from helpers import remove_unimportant_words
from helpers import flatten
import spacy
import multiprocessing
import itertools
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

# Argumentos del script
#m =  sys.argv[1]
m = "word"

nlp = spacy.load('es_core_news_md')

################
# CARGAR DATOS #
################

# Cargar centroides de las frases
with open("/home/klaus/discursos_politicos/data/centroids_text", "rb") as fp:
  centroids_text = pickle.load(fp)

with open("/home/klaus/discursos_politicos/data/centroids_sentence", "rb") as fp:
  centroids_sentence = pickle.load(fp)


# Cargar vectores cognitivo y afectivo
with open("/home/klaus/discursos_politicos/data/cognitive_vector", "rb") as fp:
  cognitive_vector = pickle.load(fp)

with open("/home/klaus/discursos_politicos/data/affective_vector", "rb") as fp:
  affective_vector = pickle.load(fp)

# Cargar frases tokenizadas
with open("/home/klaus/discursos_politicos/data/tokenization", "rb") as fp:
  tokenized = pickle.load(fp)

# Cargar frases originales
with open("/home/klaus/discursos_politicos/data/original_sentences", "rb") as fp:
  original_text = pickle.load(fp)

# Cargar listados de palabras afectivas y cognitivas
with open("/home/klaus/discursos_politicos/data/df_affective_final", "rb") as fp:
  df_affective_final = pickle.load(fp)

with open("/home/klaus/discursos_politicos/data/df_cognitive_final", "rb") as fp:
  df_cognitive_final = pickle.load(fp)

# Cargar datos originales
df =  pd.read_feather("/home/klaus/discursos_politicos/data/edicion_inicial_light.feather")

####################
# SIMILITUD COSENO #
####################

# Calcular la distancia coseno que existe entre cada frase con los polos cognitivo y afectivo. 000000
# Se crea una medida sintética que muestra hacia qué polaridad está inclinada cada frase
polarity =  [[ [(get_cosine(phrase, affective_vector) + 0.3 ) / (get_cosine(phrase, cognitive_vector) + 0.3 )  ]
              for phrase in text] for text in centroids_sentence]
cosine =  [[ [(get_cosine(phrase, affective_vector)  ) , (get_cosine(phrase, cognitive_vector)  )  ]  for phrase in text] 
           for text in centroids_sentence]
cosine[10]
polarity[10]
original_text[10]
tokenized[10]
polarity_text =  [(get_cosine(text, affective_vector) + 0.3 ) / (get_cosine(text, cognitive_vector) + 0.3 )   for text in centroids_text]

# Se destruyen los textos originales y se construyen listas de frases. Esto se hace para la medida construida de polaridad y para 
# las frases tokenizadas
flat_list_cos = flatten(polarity)
flat_list_sentences = flatten(original_text)
flat_list_cosine = flatten(cosine)


# Se agrega un identificador del texto original al que pertnece cada oración tokenizada 
flat_list_tokens = [[  sublist + [number]  for sublist in text ]  for number, text in enumerate(tokenized )]
flat_list_tokens = flatten(flat_list_tokens)

# Identificar si quedan números negativos. La mayoría debería desaparecer con la constante 0.5 que se suma arroba y abajo
negativos = [i for i in flat_list_cos if i[0] < 0 ]

len(negativos)
len(flat_list_tokens)
len(flat_list_cos)
len(flat_list_sentences)
len(flat_list_cosine)


# Unir los valores de coseno con los palabras de su correspondiente frase tokenizada y frase original
cos_words = []
for i in range(0, len(flat_list_tokens)):
  cos_words.append([flat_list_cos[i], flat_list_tokens[i][0:-1], flat_list_tokens[i][-1], flat_list_sentences[i], flat_list_cosine[i]])
len(cos_words)  

cos_words_text = [[score, number, flatten(tokenized[number]) ] for number, score in enumerate(polarity_text) ]

###########################
# ESTADÍSTICA DESCRIPTIVA #
###########################

# Generar histograma con los datos a nivel de frases
x = [i[0] for i in cos_words if len(i[1]) >= 8 and i[0][0] <= 2 ]
x = flatten(x)
np.mean(x)
np.median(x)
np.max(x)
np.min(x)
sample_score = np.random.choice(x, size=80000, replace=False)
n_bins = 100
plt.hist(x, bins = n_bins)
plt.savefig('scripts/reportes/histograma.png')
plt.figure().clear()

# Generar histograma con datos creados a partir de textos completos
x = polarity_text
np.mean(x)
np.median(x)
np.max(x)
np.min(x)
#sample_score = np.random.choice(x, size=50000, replace=False)
n_bins = 100
plt.hist(x, bins = n_bins)
plt.savefig('scripts/reportes/histograma_text.png')
plt.figure().clear()


########################
# DESCRIPCIÓN TEMPORAL #
########################
years = df.anio

# Crear un diccionario que contendrá todos los valores de coseno por año
years_dic = {int(k): [] for k in np.unique(years)  }

for cos in cos_words:
  if len(cos[1]) >= 5: 
    id_text = cos[2]
    year = years[id_text]
    years_dic[year].append(cos[0])
  
mean_per_year = {year:np.mean(values) for (year, values) in years_dic.items()}

x = list(mean_per_year.keys())
y = list(mean_per_year.values())
x.insert(9, 1974)
y.insert(9, np.nan)

plt.plot(x, y, marker = 'o')
plt.title("Frases")
plt.xticks(np.arange(1965, 2022, 5))
plt.tick_params(axis='x', rotation=70, labelsize=8)
plt.tick_params(axis='y', labelsize=8)
plt.savefig('scripts/reportes/time_sentence.png')
plt.figure().clear()

##########################
# Dejar todas las frases #
##########################

# Crear un diccionario que contendrá todos los valores de coseno por año
years_dic = {int(k): [] for k in np.unique(years)  }

for cos in cos_words_text:
  if len(cos[2]) >= 3: 
    id_text = cos[1]
    year = years[id_text]
    years_dic[year].append(cos[0])


mean_per_year = {year:np.mean(values) for (year, values) in years_dic.items()}


x = list(mean_per_year.keys())
y = list(mean_per_year.values())
x.insert(9, 1974)
y.insert(9, np.nan)

plt.plot(x, y, marker = 'o')
plt.title("Considera textos")
plt.xticks(np.arange(1965, 2022, 2))
plt.tick_params(axis='x', rotation=70, labelsize=8)
plt.tick_params(axis='y', labelsize=8)
plt.axvline(x = 1973, color = 'b', label = 'axvline - full height')
#plt.set_size_inches(18.5, 10.5)0
plt.savefig('scripts/reportes/time_text.png')
plt.figure().clear()

plt.show()

