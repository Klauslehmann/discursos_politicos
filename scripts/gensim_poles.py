import gensim 
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import re
import spacy
nlp = spacy.load('es_core_news_md')



#wordvectors2 = FastText.load_fasttext_format('data/embeddings-s-model.bin') 
wordvectors2 = load_facebook_model('data/embeddings-s-model.bin') 


##########################
# GENERAR LAS CATEGORÍAS #
##########################

# Guardar las categorías en una lista separada
categorias = []
with open('data/LIWC2001_Spanish.dic', 'r', encoding="latin-1") as reader:
    row = 0
    for line in reader.readlines():
      categorias.append(line)
      #print(line, end='')   
      row = row + 1
      if row > 68:
        break

clean_categories = [s.replace('\t','').replace('\n','') for s in categorias if s != "%\n"] 
#print(clean_categories)

codes = [re.findall(r'\d+', s) for s in clean_categories]
#print(codes)


clean_categories2 = [re.sub(r'[0-9]','', s) for s in clean_categories] 
#print(clean_categories2)

#len(codes)
#len(clean_categories2)

zip_iterator = zip(clean_categories2, codes)
categories_dic = dict(zip_iterator)

#print(categories_dic)

# Crear un diccionario que contiene las categorias asociadas a cada palabra. Una palabra puede 
# tener más de una categoría


##################
# TRABAJAR POLOS #
##################


file = open('data/LIWC2001_Spanish.dic', 'r', encoding="latin-1")
        
# lines to print
starting_point = 69  
# loop over lines in a file
words = [] 
for pos, l_num in enumerate(file):
    # check if the line number is specified in the lines to read array
    if pos > starting_point:
        codes = [re.findall(r'\d+', s) for s in l_num]
        words.append(l_num)
       
# Limpiar las líneas para que solo quede la palabra y los cóigos asociados a la palabra
raw_words = [s.replace('\n','') for s in words]
raw_words = [re.sub('\*','', n) for n in raw_words]
raw_words = list(dict.fromkeys(raw_words))

# Dejar solo la palabra de cada línea
clean_words = [re.findall(r'[a-zA-ZáéíóúñÑüÜq]*', s) for s in raw_words]
clean_words =  [[n for n in num if n] for num in clean_words]  
clean_words_flat = [item for l in clean_words for item in l]
clean_words_flat = [n for n in clean_words_flat if n]

# Extraer los códigos asociados a cada palabra
numbers = [re.findall(r'[0-9]*', s) for s in raw_words]
clean_numbers = [[n for n in num if n] for num in numbers]
clean_numbers = [n for n in clean_numbers if n]

# Eliminamos de ambas listas (palabras y códigos) los índices de las palabras que están repetidas.
duplicates = [99, 99]
while len(duplicates) > 1: 
  duplicates = [idx for idx, val in enumerate(clean_words_flat) if val in clean_words_flat[:idx]]
  del clean_words_flat[duplicates[0]]
  del clean_numbers[duplicates[0]]

# Revisar largo de las listas para chequear que esté todo en orden
# len(clean_numbers)
# len(clean_words_flat)
# len(clean_words)

# Generar un diccionario que contiene todos los códigos asociados a una palabra
zip_iterator = zip(clean_words_flat, clean_numbers)
words_with_codes = dict(zip_iterator)
#len(words_with_codes)

# Encontrar las palabras que están dentro de las categorías de interés
cognitive = ["20", "21", "22", "23", "24", "25", "26", "44", "45"]
affective = ["12", "13", "16", "17", "18", "19"]

affective_words = [key  for key, value in words_with_codes.items() if set(affective) & set(value)]
cognitive_words = [key  for key, value in words_with_codes.items() if set(cognitive) & set(value)]

# len(affective_words)
# len(cognitive_words)

# Dejar solo los adjetivos, sustantivos y verbos, según la metodología del paper 

# Buscar etiqueta pos para cada una de las palabras de la lista affective 
pos_affective = []
for word in affective_words:
  document = nlp(word)
  for token in document:
    pos_affective.append((word, token.lemma_, token.pos_)) 

pos_cognitive = []
for word in cognitive_words:
  document = nlp(word)
  for token in document:
    pos_cognitive.append((word, token.lemma_, token.pos_)) 

    
# len(pos_affective)
# len(pos_cognitive)

# Dejar solo lo que sea sustantivo, adjetivo o verbo
#print(pos_cognitive[0:10])

filter_affective = [word[1] for word in pos_affective if word[2] == "ADJ" or word[2] == "NOUN" or word[2] == "VERB" ]
filter_cognitive = [word[1] for word in pos_cognitive if word[2] == "ADJ" or word[2] == "NOUN" or word[2] == "VERB" ]

# Dejar solo las palabras que pasaron por el stemming
# Esto reduce de manera importante el número de palabras
filter_affective2 = list(dict.fromkeys(filter_affective))
filter_cognitive2 = list(dict.fromkeys(filter_cognitive))


##########################
# Trabajo con embeddings #
##########################

# Buscar el vector para cada una de las palabras 
affective_vectors_list = [wordvectors2.wv[word] for word in filter_affective2]
cognitive_vectors_list = [wordvectors2.wv[word] for word in filter_cognitive2]


# COnstruir un diccionario que una las palabras con dus respectivos vectores
zip_iterator = zip(filter_affective2, affective_vectors_list)
word_vector_affective = dict(zip_iterator)

zip_iterator = zip(filter_cognitive2, cognitive_vectors_list)
word_vector_cognitive = dict(zip_iterator)

# Construir un diccionario con las palabras que fueron encontradas 
affective_vectors_dic = {word:vector for (word, vector) in word_vector_affective.items() if sum(list(map(lambda x:pow(x,2), vector))) != 0}
cognitive_vectors_dic = {word:vector for (word, vector) in word_vector_cognitive.items() if sum(list(map(lambda x:pow(x,2), vector))) != 0}


# Calcular el centroide del espacio de vectores
affective_vectors_array = np.asarray([value for value in affective_vectors_dic.values()])
centroid_affective = np.mean(affective_vectors_array, axis=0)

cognitive_vectors_array = np.asarray([value for value in cognitive_vectors_dic.values()])
centroid_cognitive = np.mean(cognitive_vectors_array, axis=0)


# Calcular la distacia coseno
def get_cosine(vector1, vector2):
  result = dot(vector1, vector2)/(norm(vector1)*norm(vector2))
  return(result)
  
affective_cosines_dic =  {word:get_cosine(vector, centroid_affective) for (word, vector) in affective_vectors_dic.items()}
cognitive_cosines_dic =  {word:get_cosine(vector, centroid_cognitive) for (word, vector) in cognitive_vectors_dic.items()}

# Excluir las palabras que están más alejadas del centroide
data_affective = {"word" : [val for val in affective_cosines_dic.keys()], "cos" : [val for val in affective_cosines_dic.values()]}
data_cognitive = {"word" : [val for val in cognitive_cosines_dic.keys()], "cos" : [val for val in cognitive_cosines_dic.values()]}

df_affective = pd.DataFrame(data_affective)
df_affective = df_affective.sort_values(by=['cos'])

df_cognitive = pd.DataFrame(data_cognitive)
df_cognitive = df_cognitive.sort_values(by=['cos'])

drop_rows = round(df_affective.shape[0] / 4) 
df_affective_final = df_affective[drop_rows:df_affective.shape[0]]

drop_rows = round(df_cognitive.shape[0] / 4) 
df_cognitive_final = df_cognitive[drop_rows:df_cognitive.shape[0]]


################################
# Recolectar datos del proceso #
################################

start_affective =  len(affective_words)
start_cognitive = len(cognitive_words)

pos_step_affective =  len(filter_affective)
pos_step_cognitive = len(filter_cognitive)

stemm_step_affective =  len(filter_affective2)
stemm_step_cognitive = len(filter_cognitive2)

emb_step_affective = len(affective_vectors_dic)
emb_step_cognitive = len(cognitive_vectors_dic)

centroid_step_affective =  len(df_affective_final.index)
centroid_step_cognitive =  len(df_cognitive_final.index)

###########################################
# Construir el vector para cada polaridad #
###########################################

affective_vectors_list = [wordvectors2.wv[word] for word in df_affective.word]
affective_vectors_array = np.asarray(affective_vectors_list)
affective_vector = np.mean(affective_vectors_array, axis=0)

cognitive_vectors_list = [wordvectors2.wv[word] for word in df_cognitive]
cognitive_vectors_array = np.asarray(cognitive_vectors_list)
cognitive_vector = np.mean(cognitive_vectors_array, axis=0)




# wordvectors2.wv.most_similar_cosmul(positive=['rey','mujer'],negative=['hombre'])
# wordvectors2.wv.most_similar(positive=['rey','mujer'],negative=['hombre'])



