import sys
sys.path.append('scripts/')
from gensim.models.fasttext import load_facebook_model
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import re
import spacy
import pickle
import sys
from helpers import filter_important_words
from helpers import create_pole
from helpers import remove_rows
from helpers import get_cos_percentage


# Argumentos del script
#m =  sys.argv[1]
m = "word"
# Insumos
nlp = spacy.load('es_core_news_md')
wordvectors = load_facebook_model('/home/klaus/discursos_politicos/data/embeddings-m-model.bin') 

##########################
# GENERAR LAS CATEGORÍAS #
##########################

# Guardar las categorías en una lista separada
categorias = []
with open('/home/klaus/discursos_politicos/data/LIWC2001_Spanish.dic', 'r', encoding="latin-1") as reader:
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



##################
# TRABAJAR POLOS #
##################


file = open('/home/klaus/discursos_politicos/data/LIWC2001_Spanish.dic', 'r', encoding="latin-1")
        
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

# Dejar las palabras más relevantes. Se sigue la metodología del paper. Por defecto son los sustantivos, adjetivos y verbos
filter_cognitive = filter_important_words(cognitive_words, mode = m)
filter_affective = filter_important_words(affective_words, mode = m)

# Dejar solo las palabras que pasaron por el stemming
# Esto reduce de manera importante el número de palabras
filter_affective2 = list(dict.fromkeys(filter_affective))
filter_cognitive2 = list(dict.fromkeys(filter_cognitive))


##########################
# Trabajo con embeddings #
##########################

# Buscar el vector para cada una de las palabras 
affective_vectors_list = [wordvectors.wv[word] for word in filter_affective2]
cognitive_vectors_list = [wordvectors.wv[word] for word in filter_cognitive2]

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

#######################################################
# Eliminar palabras, según su cercanía con el centroide
#######################################################

# Relación entre porcentaje de palabras eliminadas y correlación 
valores = [i / 10 for i in range(1,10)]
correlacion_polos =  [get_cos_percentage(x, df_affective, df_cognitive, wordvectors) for x in valores ]

# Eliminar  las palabras que están más alejadas del centroide de cada uno de los polos
df_affective_final = remove_rows(df_affective, 0.8)
df_cognitive_final = remove_rows(df_cognitive, 0.8)


################################
# Recolectar datos del proceso #
################################

start_affective =  len(affective_words)
start_cognitive = len(cognitive_words)

pos_step_affective =  len(filter_affective)
pos_step_cognitive = len(filter_cognitive)

centroid_step_affective =  len(df_affective_final.index)
centroid_step_cognitive =  len(df_cognitive_final.index)

process_info =  pd.DataFrame({
    "polo": ["afectivo", "cognitivo"],
    "inicial": [start_affective , start_cognitive ],
    "pos": [pos_step_affective ,  pos_step_cognitive  ],
    "centroide": [centroid_step_affective,  centroid_step_cognitive ]
              }) 

process_info.to_csv("tesis/cuadros_tesis/filtrado_polos.csv")
df_cognitive_final.to_csv("tesis/cuadros_tesis/df_cognitive_final.csv")
df_affective_final.to_csv("tesis/cuadros_tesis/df_affective_final.csv")


#############################################
# Eliminar algunas palabras de los listados #
#############################################
#list(df_affective_final.word)
#list(df_cognitive_final.word)

#drop_cognitive = ["perdona", "testarud"]
#drop_affective = ["realidad"]

#df_cognitive_final = df_cognitive_final[df_cognitive_final["word"].str.contains(*drop_cognitive) == False ]
#df_affective_final = df_affective_final[df_affective_final["word"] != "realidad"]

###########################################
# Construir el vector para cada polaridad #
###########################################

affective_vector = create_pole(df_affective_final, wordvectors)    
cognitive_vector = create_pole(df_cognitive_final, wordvectors)



#################################
# Explorar listado de palabras #
#################################

# Chequear cuántas palabras están repetidas en los 2 polos
repeated = [cog for cog in df_cognitive_final.word if cog in list(df_affective_final.word) ]
len(repeated)

get_cosine(cognitive_vector, affective_vector)


df_affective_final.sort_values(by='cos', ascending=False)
df_cognitive_final.sort_values(by='cos', ascending=False)


###################
# Guardar objetos #
###################


# Guardar vector cognitivo y afectivo
with open("/home/klaus/discursos_politicos/data/cognitive_vector", "wb") as fp:
  pickle.dump(cognitive_vector, fp)


with open("/home/klaus/discursos_politicos/data/affective_vector", "wb") as fp:
  pickle.dump(affective_vector, fp)

  
# Guardar listado de palabras del polo afectivo y cognitivo 
with open("/home/klaus/discursos_politicos/data/df_cognitive_final", "wb") as fp:
  pickle.dump(df_cognitive_final, fp)

with open("/home/klaus/discursos_politicos/data/df_affective_final", "wb") as fp:
  pickle.dump(df_affective_final, fp)


