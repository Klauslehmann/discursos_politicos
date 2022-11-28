import pickle
from statistics import median
import numpy as np
import pandas as pd

# Cargar textos tokenizados a nivel de párrafo
tokenized = []
for parte in range(1, 11):
    file_name = "/home/klaus/discursos_politicos/data/tokenization_parte{parte}".format(parte = parte)
    with open(file_name, "rb") as fp:
      particion = pickle.load(fp)
      tokenized.extend(particion)        


def flatten(xss):
    return [x for xs in xss for x in xs]


# Cargar datos tokenizados a nivel de frase
tokenized_phrases = []
for parte in range(1, 11):
    file_name = "/home/klaus/discursos_politicos/data/tokenization_phrases_parte{parte}".format(parte = parte)
    with open(file_name, "rb") as fp:
      particion = pickle.load(fp)
      tokenized_phrases.extend(particion)        


############################################################
# Obtener estadística descriptiva de los textos preprocesados
#############################################################
lista_parrafos = flatten(tokenized)
palabras_parrafo = [len(par ) for par in lista_parrafos]

# Estadísticas generales del dataset
n_intervenciones = len(tokenized )
n_parrafos = len(lista_parrafos)
n_palabras = sum(palabras_parrafo )
media_parrafo_interv = n_parrafos / n_intervenciones 
media_palabras_interv = n_palabras / n_intervenciones 

dataset_stats = pd.DataFrame(
    {"n_intervenciones": [n_intervenciones],
     "n_parrafos": [n_parrafos],
     "n_palabras": [n_palabras],
     "media_parrafo_interv": [media_parrafo_interv],
     "media_palabras_interv": [media_palabras_interv]
     }
    )


# Estadísticas a nivel de párrafo
parrafo_mean = np.mean(palabras_parrafo )
parrafo_min = min(palabras_parrafo )
parrafo_max = max(palabras_parrafo )
parrafo_median = median(palabras_parrafo )


parrafo_stats =  pd.DataFrame(
    {"media": [parrafo_mean],
     "mediana": [parrafo_median],
     "min": [parrafo_min],
     "max": [parrafo_max]
     
     }
    )

palabras_parrafo = pd.DataFrame(
    {"n_palabras": palabras_parrafo
     }
    )

# Guardar información de estadísticas generales del dataset
dataset_stats.to_csv("tesis/cuadros_tesis/datasets_stats.csv", index = False )

# Guardar información sobre el largo de los párrafos
palabras_parrafo .to_csv("tesis/cuadros_tesis/largo_parrafos.csv", index = False )




########################################################################
# Obtener estadística descriptiva de los textos preprocesados en frases
##########################################################################
lista_parrafos = flatten(tokenized_phrases)
palabras_parrafo = [len(par ) for par in lista_parrafos]

# Estadísticas generales del dataset
n_intervenciones = len(tokenized_phrases )
n_parrafos = len(lista_parrafos)
n_palabras = sum(palabras_parrafo )
media_parrafo_interv = n_parrafos / n_intervenciones 
media_palabras_interv = n_palabras / n_intervenciones 

dataset_stats = pd.DataFrame(
    {"n_intervenciones": [n_intervenciones],
     "n_parrafos": [n_parrafos],
     "n_palabras": [n_palabras],
     "media_parrafo_interv": [media_parrafo_interv],
     "media_palabras_interv": [media_palabras_interv]
     }
    )


# Estadísticas a nivel de párrafo
parrafo_mean = np.mean(palabras_parrafo )
parrafo_min = min(palabras_parrafo )
parrafo_max = max(palabras_parrafo )
parrafo_median = median(palabras_parrafo )


parrafo_stats_phrases =  pd.DataFrame(
    {"media": [parrafo_mean],
     "mediana": [parrafo_median],
     "min": [parrafo_min],
     "max": [parrafo_max]
     
     }
    )

palabras_parrafo = pd.DataFrame(
    {"n_palabras": palabras_parrafo
     }
    )

# Guardar información de estadísticas generales del dataset
dataset_stats.to_csv("tesis/cuadros_tesis/datasets_stats_phrases.csv", index = False )

# Guardar información sobre el largo de los párrafos
palabras_parrafo .to_csv("tesis/cuadros_tesis/largo_parrafos_phrases.csv", index = False )



import pandas as pd
df = pd.read_feather("data/score_filtered.feather" )


df["nwords"] = [len(text.split()) for text in df.text]

df[df["nwords"] >= 10].sort_values(by = ["score"], ascending=False)["text"][0:10]

