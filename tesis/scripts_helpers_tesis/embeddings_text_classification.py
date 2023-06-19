import pickle
import sys
sys.path.append('scripts/')
from helpers import get_cosine
from helpers import flatten
import spacy
from sklearn.decomposition import PCA
import numpy as np
#import matplotlib as plt
import pandas as pd
import matplotlib.pyplot as plt



# Argumentos del script
#m =  sys.argv[1]
m = "word"

nlp = spacy.load('es_core_news_md')

################
# CARGAR DATOS #
################

centroids_sentence = []
for parte in range(1, 11):
    file_name = "/home/klaus/discursos_politicos/data/centroids_sentence_parte{parte}".format(parte = parte)
    with open(file_name, "rb") as fp:
      particion = pickle.load(fp)
      centroids_sentence.extend(particion)        

tokenized = []
for parte in range(1, 11):
    file_name = "/home/klaus/discursos_politicos/data/tokenization_parte{parte}".format(parte = parte)
    with open(file_name, "rb") as fp:
      particion = pickle.load(fp)
      tokenized.extend(particion)        

#original_text = []
#for parte in range(1, 11):
#    file_name = "/home/klaus/discursos_politicos/data/original_sentences_parte{parte}".format(parte = parte)
#    with open(file_name, "rb") as fp:
#      particion = pickle.load(fp)
#      original_text.extend(particion)        
    
# Cargar vectores cognitivo y afectivo
with open("/home/klaus/discursos_politicos/data/cognitive_vector", "rb") as fp:
  cognitive_vector = pickle.load(fp)

with open("/home/klaus/discursos_politicos/data/affective_vector", "rb") as fp:
  affective_vector = pickle.load(fp)
    
# Cargar listados de palabras afectivas y cognitivas
with open("/home/klaus/discursos_politicos/data/df_affective_final", "rb") as fp:
  df_affective_final = pickle.load(fp)

with open("/home/klaus/discursos_politicos/data/df_cognitive_final", "rb") as fp:
  df_cognitive_final = pickle.load(fp)
    

##############################
# GUARDAR VECTORES EN FEATHER
##############################

d = {'cognitive': cognitive_vector, 'affective': affective_vector}
df_vectors =  pd.DataFrame(data=d)

df_vectors .to_feather("tesis/cuadros_tesis/polos_df.feather")

####################
# SIMILITUD COSENO #
####################

# Calcular la distancia coseno que existe entre cada frase con los polos cognitivo y afectivo. 000000
# Se crea una medida sintética que muestra hacia qué polaridad está inclinada cada frase
polarity =  [[ [(get_cosine(phrase, affective_vector) + 0.3 ) / (get_cosine(phrase, cognitive_vector) + 0.3 )  ]
              for phrase in text] for text in centroids_sentence]

# Se destruyen los textos originales y se construyen listas de frases. Esto se hace para la medida construida de polaridad y para 
# las frases tokenizadas
flat_list_polarity = flatten(polarity)
flat_list_vectors = flatten(centroids_sentence)

##################################
# CONSEGUIR COGNITIVAS Y AFECTIVAS
##################################

# Crear lista que contiene polaridad en el primer elemento y vector en el segundo
polarity_vector =  [[flat_list_polarity[i], flat_list_vectors [i] ] for i in range(len(flat_list_polarity)) ]


# Ordenar de menor a mayor los puntajes
ordenados = sorted(polarity_vector, key=lambda tup: tup[0])

# Se seleccionanan los 50 más afectivos con los 50 más cognitivos
n = 10000

cognitivos = ordenados[0:n]
afectivos = ordenados[-n:] 


# Obtener los vectores de cada polaridad
vec_cognitivos = [list(i[1])  for i in cognitivos]
vec_afectivos  = [list(i[1]) for i in afectivos]


seen = []
unique_vec_cognitivos = []
i = 0
for vector in list(vec_cognitivos) :    
    if vector not in seen :
        unique_vec_cognitivos .append(vector)
        i = i + 1 
    seen.append(vector) 

seen = []
unique_vec_afectivo= []
i = 0
for vector in list(vec_afectivos) :    
    if vector not in seen :
        unique_vec_afectivo.append(vector)
        i = i + 1 
    seen.append(vector) 

m = 3000
unique_vec_cognitivos = unique_vec_cognitivos [0:m]
unique_vec_afectivo = unique_vec_afectivo[0:m]


############
# HACER PCA
############

unique_vec_cognitivos .extend(unique_vec_afectivo) 


# Generar la matriz con los vectores de palabras para el ejemplo 
word_vectors = np.array(unique_vec_cognitivos )

# Obtener la varianza explicada de las 2 primeras componentes
pca_function = PCA(n_components=2)
principalComponents = pca_function.fit_transform(word_vectors)
variance =  pca_function.explained_variance_ratio_
 

# Extraer las 2 primeras componentes para hacer el gráfico
pca = PCA().fit_transform(word_vectors)
twodim = pca[:,:2]

label_cognitivo = ["cognitivo" for i in range(m)]
label_afectivo= ["afectivo" for i in range(m)]

label_cognitivo.extend(label_afectivo) 


pca_example = pd.DataFrame({
     "polo": label_cognitivo,
     "d1": twodim[:,0], 
     "d2": twodim[:,1] })
 
pca_example.groupby("polo").size()

# Graficar
fig, ax = plt.subplots()
x = pca_example.d1
y = pca_example.d2

colors = {'cognitivo':'red', 'afectivo':'green'}
ax.scatter(pca_example.d1, pca_example.d2, c = pca_example['polo'].map(colors)) # 
fig

# Guardar tabla de datos

pca_example.to_csv("tesis/cuadros_tesis/scatter_example_polarity.csv")
