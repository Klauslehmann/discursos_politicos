import pickle
import pandas as pd
import sys
sys.path.append('scripts/')
from helpers import get_cosine
from helpers import flatten
import spacy
import numpy as np
from Levenshtein import distance

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

original_text = []
for parte in range(1, 11):
    file_name = "/home/klaus/discursos_politicos/data/original_sentences_parte{parte}".format(parte = parte)
    with open(file_name, "rb") as fp:
      particion = pickle.load(fp)
      original_text.extend(particion)        
    
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


############################
# CONSTRUIR TABLA DE DATOS #
############################

# Limpiar memoria
del centroids_sentence, tokenized, original_text, flat_list_tokens, flat_list_cos, flat_list_sentences, flat_list_cosine, particion, polarity, cosine
del affective_vector, cognitive_vector
del df_affective_final, df_cognitive_final, fp

# Diccionario con los datos de las frases
data = {'id_speech': [i[2] for i in cos_words],
        'score': [i[0][0] for i in cos_words],
        'text': [i[3] for i in cos_words],
        'text_edit': [ " ".join(i[1])  for i in cos_words ], 
        'cos_affect': [i[4][0] for i in cos_words],
        'cos_cognitive': [i[4][1] for i in cos_words],
        'n_words' : [len(i[1]) for i in cos_words]
        }

# Limpiar memoria
del  cos_words

# Cargar datos originales
df =  pd.read_feather("/home/klaus/discursos_politicos/data/edicion_inicial.feather")

# Convertir a data frame
df_phrases = pd.DataFrame(data)

# Crear identificador para la tabla que contiene información de los diputados. Este identificador corresponde a cada una de las intervenciones
df['id_speech'] = np.arange(len(df))

# Construir tabla con las columnas relevantes
df_full = pd.merge(df[["id_speech", "polo", "fecha", "texto_dep", "anio", "partido", "nombre", "edad_actual", "sexo", "periodo_legislativo"]], df_phrases, on='id_speech')

# Crear un id correlativo para cada una de las frases
df_full ['id_phrase'] = np.arange(len(df_full))

# Excluir los textos con menos de 4 tokens
df_filtrado = df_full[df_full["n_words"] >= 4].reset_index()





#################
# GUARDAR DATOS #
#################

# Se guarda el archivo full en esta sección para liberar un poco de memoria
del data, df_phrases
del df_full["texto_dep"]  # borrar la información de textos, porque hace que el archivo pese demasiado


df_full.to_feather("data/score_full.feather")
del df_full
del df_filtrado["texto_dep"]
del df

# Agregar información que indica si el político era senador o diputado
diputado_senador = pd.read_feather("data/periodos_parlamentarios.feather")
diputado_senador = diputado_senador [["name", "role", "anio"]]
diputado_senador = diputado_senador.rename(columns={"name": "nombre_edit"})
diputado_senador["role"]  = np.where(diputado_senador["role"] == "diputada", "diputado", diputado_senador["role"])


# Se editan nombres en la tabla de biografías para que hagan match con la tabla de intervenciones. Se parte con una estrategia 
# carretera y luego se hace algo más sistemático

diputado_senador["nombre_edit"] = np.where(diputado_senador["nombre_edit"] == "adriana muñoz d'albora",  
                                           "adriana muñoz d' albora", diputado_senador["nombre_edit"])

diputado_senador["nombre_edit"] = np.where(diputado_senador["nombre_edit"] == "alejandro javier bernales maldonado",  
                                           "alejandro bernales maldonado", diputado_senador["nombre_edit"])

diputado_senador["nombre_edit"] = np.where(diputado_senador["nombre_edit"] == "alejandro rené eleodoro guillier álvarez",  
                                           "alejandro guillier álvarez", diputado_senador["nombre_edit"])


diputado_senador["nombre_edit"] = np.where(diputado_senador["nombre_edit"] == "alexis edgardo sepúlveda soto",  
                                           "alexis sepúlveda soto", diputado_senador["nombre_edit"])


diputado_senador["nombre_edit"] = np.where(diputado_senador["nombre_edit"] == "alfonso arturo de urresti longton",  
                                           "alfonso de urresti longton", diputado_senador["nombre_edit"])


diputado_senador["nombre_edit"] = np.where(diputado_senador["nombre_edit"] == "alvaro erich schnake silva",  
                                           "alvaro erick schnake silva", diputado_senador["nombre_edit"])

diputado_senador["nombre_edit"] = np.where(diputado_senador["nombre_edit"] == "amaro jamil jesús labra sepúlveda",  
                                           "amaro labra sepúlveda", diputado_senador["nombre_edit"])


diputado_senador["nombre_edit"] = np.where(diputado_senador["nombre_edit"] == "irle andrea parra sauterel",  
                                           "andrea parra sauterel", diputado_senador["nombre_edit"])



diputado_senador["nombre_edit"] = np.where(diputado_senador["nombre_edit"] == "josé sergio andrés molina magofke",  
                                           "andrés molina magofke", diputado_senador["nombre_edit"])


 
diputado_senador["nombre_edit"] = np.where(diputado_senador["nombre_edit"] == "beltrán urenda zegers",  
                                           "beltran urenda zegers", diputado_senador["nombre_edit"])




# Pegar datos de biografía a la tabla principal.
# Este merge hace que se dupliquen algunas filas. Esto ocurre debido a que hay personas en la tabla de periodos
# que pasan de un cargo de diputado a un cargo de senador en un mismo año. Se toma la decisión de mantener ambos
df_filtrado['nombre_edit'] = df_filtrado["nombre"].str.lower()
df_filtrado = pd.merge(df_filtrado , diputado_senador, how="left",  on= ['anio', "nombre_edit"] )


# Luego de hacer el merge, una parte importante no hace match.
# Se implementa una estrategia basada en similitud de textos para hacer el match. La idea es recorrer todos los nombres
# y seleccionar al que más se parece, pero para evitar un sobreprocesamiento, se hacen algunos filtros
# Primero, se selecciona a todos los políticos que están dentro del periodo que se está estudiando. 
# En segundo lugar, se borran todos los políticos que ya hicieron match.
nombres_con_match=df_filtrado[~df_filtrado["role"].isnull()].groupby( ["nombre_edit"] ).first().reset_index()[["nombre_edit"]]
politicos_periodo = diputado_senador[diputado_senador["anio"] >= 1965  ].groupby("nombre_edit").first().reset_index()
candidatos_a_match = politicos_periodo[~politicos_periodo['nombre_edit'].isin(nombres_con_match["nombre_edit"])]
no_match=df_filtrado[df_filtrado["role"].isnull()]
nombres_sin_match = no_match.groupby( ["nombre_edit"] ).first().reset_index()[["nombre_edit"]]




def find_politician(politician_name, list_of_names):
    min_dist = 99999
    for name in list_of_names:    
        dist = distance(name, politician_name)
        if dist < min_dist:
            min_dist = dist
            min_name = name
    name_list = politician_name.split()
    min_name_list = min_name.split()
    counter = 0
    # contar número de coincidencia de palabras    
    for n in name_list:
        for n2 in min_name_list:
            if n == n2:
                counter = counter + 1
    # Identificar personas con apellido repetido
    if len(name_list) > len(np.unique(name_list)):
        apellido_repetido = True
    else:
        apellido_repetido = False    
    #return apellido_repetido, counter 
    # Si al menos hay 2 coincidencias y el apellido no está repetido, se valida el nombre
    if (counter >= 2) & (apellido_repetido == False):
        return min_name
    # Si el apellido está repetido, se exigen 3 coincidencias
    elif (counter >= 3) & (apellido_repetido == True):
        print(apellido_repetido , counter)
        return min_name
    else:
        return None 

#find_politician("adrian alvarez alvarez",  candidatos_a_match["nombre_edit"])


closest_name = [find_politician(name, candidatos_a_match["nombre_edit"])  for name in nombres_sin_match ["nombre_edit"]]    

nombres_sin_match ["closest_name"] = closest_name
nombres_sin_match = nombres_sin_match [~nombres_sin_match ["closest_name"].isnull()]
diputado_senador2 = diputado_senador.merge(nombres_sin_match, how = "right", 
                                           left_on='nombre_edit', 
                                           right_on='closest_name'  )

# Editar nombres en la tabla de biografías, para que calcen con la tabla de intervenciones 
diputado_senador2 = diputado_senador2 [["nombre_edit_y", "role", "anio"]]
diputado_senador2 = diputado_senador2.rename(columns={"nombre_edit_y": "nombre_edit"})


# Se vuelve a pegar la información sobre políticos con los nombres ya corregidos
df_filtrado = df_filtrado.merge(diputado_senador2, how = "left", on = ["nombre_edit", "anio"])
df_filtrado["role"] = np.where(~df_filtrado["role_x"].isnull(), df_filtrado["role_x"], df_filtrado["role_y"])
df_filtrado = df_filtrado.loc[:, ~df_filtrado.columns.isin(["role_x", "role_y"])]
df_filtrado .to_feather("data/score_filtered.feather")




###################
# LIMPIAR MEMORIA #
###################

from IPython import get_ipython
get_ipython().magic('reset -sf')


###############
# CREAR TÓPICOS 
###############


# =============================================================================
# # Cargar el modelo para encontrar tópicos
# classifier = pipeline("zero-shot-classification", 
#                        model="Recognai/zeroshot_selectra_medium")
# 
# 
# # Defunir tópicos 
# topics = ["salud", "educación", "deporte", 
#          "medioambiente", "impuestos", 
#          "cultura", "pensiones", 
#          "sindicalismo", "transporte", "familia", "aborto"]
# 
# # Seleccionar una muestra para crear los tópicos
# n = df_filtrado.shape[0]
# n = 300000
# tx  = df_filtrado.text.sample(n = n, random_state = 1,)
# 
# # Clasificar cada texto con un tópico
# tic = time.time()
# topic = [classifier(text, candidate_labels = topics ) 
#          for text in tx]
# toc = time.time()
# print('Done in {:.4f} seconds'.format(toc-tic))
# 
# 
# # Rescatar la etiqueta y el puntaje asignado por el modelo a la etiqueta
# labels = [t["labels"][0] for t in topic]
# score_topic = [t["scores"][0] for t in topic]
# indices = tx.index
# 
# # Constuir dataframe con un subconjunto de textos clasificados con tópico 
# dic = {"id_phrase": df_filtrado.id_phrase[indices],
#        "id_original_text": df_filtrado.id_speech[indices],
#        "name": df_filtrado.nombre[indices],
#        "text": tx, 
#        "topic": labels,
#        "score_topic": score_topic,
#        "score":df_filtrado.score[indices],
#        "pole": df_filtrado.polo[indices],
#        "emotion": df_filtrado.cos_affect[indices], 
#        "cognition": df_filtrado.cos_cognitive[indices],
#        "age": df_filtrado.edad_actual[indices],
#        "year": df_filtrado.anio[indices],
#        "sex": df_filtrado.sexo[indices]
#        
#        }
#  
# df_topics= pd.DataFrame(dic ).reset_index()
# 
# # Verificar que las filas no se hayan corrido entre full y la selección para tópicos
# df_topics[df_topics["id_phrase"]== 20715].text_dep 
# df_filtrado[df_filtrado["id_phrase"]== 20715].text 
# 
# # Verificar que la tabla de tópicos corresponda a los textos originales
# id_test = df_topics[df_topics["id_phrase"] == 20715].id_original_text
# 
# test_text1 = df_topics[df_topics["id_phrase"]== 20715].text_dep
# test_text2 =  df[df["id_speech"] == int(id_test) ].texto_dep 
# 
# 
# #################
# # GUARDAR DATOS #
# #################
# del df_filtrado["texto_dep"]
# df_filtrado .to_feather("data/score_filtered.feather")
# df_filtrado .to_csv("data/score_filtered.csv")
# df_topics.to_feather("data/topics_test_paragraph.feather")
# 
# =============================================================================

