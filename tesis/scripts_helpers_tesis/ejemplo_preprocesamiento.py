
import pickle
import pandas as pd

# Cargar datos editados
df =  pd.read_feather("/home/klaus/discursos_politicos/data/edicion_inicial.feather")

file_name = "/home/klaus/discursos_politicos/data/tokenization_parte{parte}".format(parte = 1)

tokenized = []
with open(file_name, "rb") as fp:
    particion = pickle.load(fp)
    tokenized.extend(particion)  


original_text = []
file_name = "/home/klaus/discursos_politicos/data/original_sentences_parte{parte}".format(parte = 1)
with open(file_name, "rb") as fp: 
    particion = pickle.load(fp)
    original_text.extend(particion)        

df.texto[1]
df.texto_dep[1]
original_text [1][1]
tokenized[1][1]

final = "['destaco', 'queremos', 'trabajar', 'proyectos', 'parlamentarios', 'visto', 'busca', 'proyecto', 'ministro', \
 'anticipado', 'queremos', 'aliviar', 'familias', 'materia', 'crediticia', 'compartimos', 'espíritu', \
 'quiere', 'explica', 'queramos', 'trabajar', 'proyectos', 'ley', 'empujado', 'sacado']"

original = "Lo destaco, porque queremos trabajar en los proyectos de los parlamentarios. \
Hemos visto lo que se busca con este proyecto, el ministro de Hacienda ya había anticipado que queremos aliviar \
 a las familias en materia crediticia y compartimos el espíritu de lo que se quiere. Y eso es justamente lo que explica \
 que queramos trabajar sobre los diversos proyectos de ley que ustedes han empujado y han sacado adelante."


ejemplo_preprocesamiento = pd.DataFrame({"original": [original], "final": [final]} )

ejemplo_preprocesamiento .to_csv("tesis/cuadros_tesis/ejemplo_preprocesamiento.csv")
