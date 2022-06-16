from numpy import dot
from numpy.linalg import norm
import pickle
from collections import Counter
import pandas as pd
import sys
sys.path.append('scripts/')
from helpers import get_cosine
from helpers import remove_unimportant_words
from helpers import get_words_ranking
from helpers import flatten
import spacy
from gensim.models.fasttext import load_facebook_model
import multiprocessing
import itertools
from functools import partial
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Argumentos del script
m =  sys.argv[1]
m = "word"

nlp = spacy.load('es_core_news_md')

################
# CARGAR DATOS #
################


# Cargar centroides de las frases
with open("/home/klaus/discursos_politicos/data/centroids_sentence", "rb") as fp:
  centroids_sentence = pickle.load(fp)

with open("/home/klaus/discursos_politicos/data/centroids_text", "rb") as fp:
  centroids_text = pickle.load(fp)

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

# Calcular la distancia coseno que existe entre cada frase con los polos cognitivo y afectivo. 
# Se crea una medida sintética que muestra hacia qué polaridad está inclinada cada frase
polarity =  [[ [(get_cosine(phrase, affective_vector) + 0.3) / (get_cosine(phrase, cognitive_vector) + 0.3)] for phrase in text] for text in centroids_sentence]

polarity_text =  [(get_cosine(text, affective_vector) + 0.3 ) / (get_cosine(text, cognitive_vector) + 0.3 )   for text in centroids_text]


# Se destruyen los textos originales y se construyen listas de frases. Esto se hace para la medida construida de polaridad y para 
# las frases tokenizadas
flat_list_cos = flatten(polarity)
flat_list_sentences = flatten(original_text)

# Se agrega un identificador del texto original al que pertnece cada oración tokenizada 
flat_list_tokens = [[ sentence  + [number]  for sentence in text  ] for number, text in enumerate(tokenized)]
flat_list_tokens = flatten(flat_list_tokens)

# Identificar si quedan números negativos. La mayoría debería desaparecer con la constante 0.5 que se suma arroba y abajo
negativos = [i for i in flat_list_cos if i[0] < 0 ]
len(negativos)
len(flat_list_tokens)
len(flat_list_cos)
len(flat_list_sentences)

# Unir los valores de coseno con los palabras de su correspondiente frase tokenizada y frase original
cos_words = []
for i in range(0, len(flat_list_tokens)):
  cos_words.append([flat_list_cos[i], flat_list_tokens[i][0:-1], flat_list_tokens[i][-1], flat_list_sentences[i] ])

cos_words_text = [[score, flatten(tokenized[number]), number, original_text[number]] for number, score in enumerate(polarity_text) ]

# Ordenmos de menor a mayor (valores pequeños están asociados al polo cognitivo)
cos_words.sort(key=lambda x: x[0])
cos_words_text.sort(key=lambda x: x[0])

# Frases más afectivas y cognitivas con un largo mínimo
long_phrases = [w for w in cos_words if len(w[1]) > 8]  
[x[3] for x in long_phrases[-10:-1] ] # affectiva
[x[3] for x in long_phrases[0:10] ] # cognitiva

long_phrases_text = [w for w in cos_words_text if len(w[1]) > 5]  
[x[3] for x in long_phrases_text[-3:-2] ] # afectiva
[x[3] for x in long_phrases_text[1:2] ] # cognitiva

# Dejas las palabras de las 500 frases más cognitivas
w1 = get_words_ranking(cos_words, pole = "cognitive")

# Dejas las palabras de las 500 frases más afectivas
w2 = get_words_ranking(cos_words, pole = "affective")


############################################
# Evaluar la clasificación de las palabras #
#############################################

# Cargar modelo de embeddings
wordvectors = load_facebook_model('/home/klaus/discursos_politicos/data/embeddings-s-model.bin') 

# Generar un subset para probar
texts = df.texto_dep
len(texts)

# Eliminar palabras poco interesantes 
cpus = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cpus)
clean_words = pool.map(partial(remove_unimportant_words, relevant_pos = ["NOUN", "ADJ", "VERB"], mode = m), texts)
pool.close()

# Dejar solo las palabras únicas
vocab = []
seen = set()
for text in clean_words:
  for word in text:
    if word not in seen:
      vocab.append(word)
    seen.add(word)

# Sacar las palabras que ya están dentro del diccionario de cada uno de los polos
vocab2 =  [word for word in vocab if word not in list(df_affective_final.word) and word not in list(df_cognitive_final.word) ]
len(vocab)
len(vocab2)


# Buscar distancia coseno de cada una de las palabras respecto a los 2 polos
similarity_affective = []
similarity_cognitive = []
for w in vocab2:
  vector = wordvectors.wv[w]
  similarity_affective.append( [w, get_cosine(vector, affective_vector)] )
  similarity_cognitive.append([w, get_cosine(vector, cognitive_vector)]  )

# Mostrar palabras más relevantes de cada polo
similarity_cognitive.sort(key=lambda x: x[1], reverse = True)
similarity_affective.sort(key=lambda x: x[1], reverse = True)

similarity_affective[0:30]
similarity_cognitive[0:30]

##################
# Crear gráficos #
##################

keys_list = [pair[0] for pair in similarity_affective[0:100]]
values_list = [pair[1] for pair in similarity_affective[0:100]]
zip_iterator = zip(keys_list, values_list)
weights = dict(zip_iterator)

# Generate the cloud
wc = WordCloud(width=1600, height=800)
wordcload_affect = wc.generate_from_frequencies(weights)
plt.figure(figsize=(20,10) )
plt.imshow(wordcload_affect, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
plt.savefig('scripts/reportes/wordcloud_affective.png', facecolor='k', bbox_inches='tight')



keys_list = [pair[0] for pair in similarity_cognitive[0:100]]
values_list = [pair[1] for pair in similarity_cognitive[0:100]]
zip_iterator = zip(keys_list, values_list)
weights = dict(zip_iterator)

# Generate the cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wc = WordCloud(width=1600, height=800)
wordcload_affect = wc.generate_from_frequencies(weights)
plt.figure(figsize=(20,10) )
plt.imshow(wordcload_affect, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

plt.savefig('scripts/reportes/wordcloud_cognitive.png', facecolor='k', bbox_inches='tight')

