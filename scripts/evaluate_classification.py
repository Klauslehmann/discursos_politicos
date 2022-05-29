from numpy import dot
from numpy.linalg import norm
import pickle
from collections import Counter
import pandas as pd
sys.path.append('scripts/')
from helpers import get_cosine


################
# CARGAR DATOS #
################

df =  pd.read_feather("/mnt/c/proyectos_personales/discursos_politicos/data/clean_data/edicion_inicial_light.feather")

list(df.columns)
# Cargar centroides de las frases
with open("data/centroids", "rb") as fp:
  centroids = pickle.load(fp)

# Cargar vectores cognitivo y afectivo
with open("data/cognitive_vector", "rb") as fp:
  cognitive_vector = pickle.load(fp)

with open("data/affective_vector", "rb") as fp:
  affective_vector = pickle.load(fp)

# Cargar frases tokenizadas
with open("data/tokenization", "rb") as fp:
  tokenized = pickle.load(fp)

# Cargar frases originales
with open("data/original_sentences", "rb") as fp:
  original_text = pickle.load(fp)

####################
# SIMILITUD COSENO #
####################

# Calcular la distancia coseno que existe entre cada frase con los polos cognitivo y afectivo. 
# Se crea una medida sintética que muestra hacia qué polaridad está inclinada cada frase
polarity =  [[ [(get_cosine(phrase, affective_vector) + 0.5 ) / (get_cosine(phrase, cognitive_vector) + 0.5 )  ]  for phrase in text] for text in centroids]
len(polarity)

# Se destruyen los textos originales y se construyen listas de frases. Esto se hace para la medida construida de polaridad y para 
# las frases tokenizadas
flat_list_cos = []
for sublist in polarity:
  flat_list_cos = flat_list_cos + sublist 

flat_list_tokens = []
for sublist in tokenized:
  flat_list_tokens = flat_list_tokens + sublist 

flat_list_sentences = []
for sublist in original_text:
  flat_list_sentences = flat_list_sentences + sublist 
i = 47
flat_list_sentences[i]
flat_list_tokens[i]

# Se agrega un identificador al final de cada frase, para 
flat_list_tokens = []
id_text = 0
for sublist in tokenized:
  x = sublist.copy()
  with_id = [sub + [id_text] for sub in sublist]
  flat_list_tokens = flat_list_tokens + with_id
  id_text += 1

# Identificar si quedan números negativos. La mayoría debería desaparecer con la constante 0.5 que se suma arroba y abajo
negativos = [i for i in flat_list_cos if i[0] < 0 ]

len(negativos)
len(flat_list_tokens)
len(flat_list_cos)
len(flat_list_sentences)
# Unir los valores de coseno con los palabras de su correspondiente frase tokenizada
cos_words = []
for i in range(0, len(flat_list_tokens)):
  cos_words.append([flat_list_cos[i], flat_list_tokens[i][0:-1], flat_list_tokens[i][-1] ])

# Ordenmos de menor a mayor (valores pequeños están asociados al polo cognitivo)
cos_words.sort(key=lambda x: x[0])

# Frases más afectivas y cognitivas con un largo mínimo
long_phrases = [w for w in cos_words if len(w[1]) > 5]  
long_phrases[-10:-1]
long_phrases[0:10]



# Dejar las quinientas frases más afectivas y contar frecuencia de palabras
affect_phrases =  cos_words[-500:-1]
affect_words = [x[1] for x in affect_phrases ]
affect_words = [item for sublist in affect_words for item in sublist]
word_counts_affect = Counter(affect_words)
sorted(word_counts_affect.items(), key=lambda item: item[1], reverse=True)[0:50]

# Dejas las 500 frases más cognitivas
cognitive_phrases =  cos_words[0:500]
cognitive_words = [x[1] for x in cognitive_phrases ]
cognitive_words = [item for sublist in cognitive_words for item in sublist]
word_counts_cognitive = Counter(cognitive_words)
sorted(word_counts_cognitive.items(), key=lambda item: item[1], reverse=True)[0:50]


