from numpy import dot
from numpy.linalg import norm
import pickle
from collections import Counter
import pandas as pd
import sys
sys.path.append('scripts/')
from helpers import get_cosine
from helpers import remove_unimportant_words
import spacy
from gensim.models.fasttext import load_facebook_model
import multiprocessing
import itertools
from functools import partial


# Argumentos del script
m =  sys.argv[1]


nlp = spacy.load('es_core_news_md')

################
# CARGAR DATOS #
################

# Cargar modelo de embeddings
wordvectors = load_facebook_model('/home/klaus/discursos_politicos/data/embeddings-s-model.bin') 

# Cargar centroides de las frases
with open("/home/klaus/discursos_politicos/data/centroids", "rb") as fp:
  centroids = pickle.load(fp)

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

# Se agrega un identificador del texto original al que pertnece cada oración tokenizada 
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

# Unir los valores de coseno con los palabras de su correspondiente frase tokenizada y frase original
cos_words = []
for i in range(0, len(flat_list_tokens)):
  cos_words.append([flat_list_cos[i], flat_list_tokens[i][0:-1], flat_list_tokens[i][-1], flat_list_sentences[i] ])

# Ordenmos de menor a mayor (valores pequeños están asociados al polo cognitivo)
cos_words.sort(key=lambda x: x[0])

# Frases más afectivas y cognitivas con un largo mínimo
long_phrases = [w for w in cos_words if len(w[1]) > 5]  
[x[3] for x in long_phrases[-10:-1] ] 
[x[3] for x in long_phrases[0:10] ] 


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

############################################
# Evaluar la clasificación de las palabras #
#############################################

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



keys_list = [pair[0] for pair in similarity_affective[0:100]]
values_list = [pair[1] for pair in similarity_affective[0:100]]
zip_iterator = zip(keys_list, values_list)
weights = dict(zip_iterator)

# Generate the cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wc = WordCloud(width=1600, height=800)
wordcload_affect = wc.generate_from_frequencies(weights)
plt.figure(figsize=(100,50) )
plt.imshow(wordcload_affect, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

plt.savefig('wordcloud.png', facecolor='k', bbox_inches='tight')



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

plt.savefig('wordcloud.png', facecolor='k', bbox_inches='tight')

