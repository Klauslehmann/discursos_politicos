
#del pre_process_text

import sys
sys.path.append('scripts/')
from gensim.models.fasttext import load_facebook_model
from helpers import pre_process_text
from helpers import convert_to_vec
import numpy as np
import pandas as pd
from helpers import save_list
import time
import copy

# Argumentos del script
#m =  sys.argv[1]
m = "word"

# Cargar datos editados
df =  pd.read_feather("/home/klaus/discursos_politicos/data/edicion_inicial.feather")

# Cargar modelo de embeddings
wordvectors = load_facebook_model('/home/klaus/discursos_politicos/data/embeddings-m-model.bin') 



# Dividir texto en varias partes
text_vector = copy.copy(df.texto_dep)
split_text = np.array_split(text_vector, 10)

#corrió hasta la partición 6

# Procesar texto y guardar cada parte por separado, para no colapsar la memoria
particion = 1
start_time = time.time()
for fraction in split_text:
    #pool = multiprocessing.Pool(processes=cpus)
    #tokenized_text = pool.map(partial(pre_process_text, relevant_pos = ["NOUN", "ADJ", "VERB"], mode = "word", paragraph = True), fraction )
    tokenized_text = [pre_process_text(t, paragraph=True) for t in fraction]
    tokenized = list(map(lambda x:x[0], tokenized_text)) 
    original_text = list(map(lambda x:x[1], tokenized_text))

    # Convertir cada palabra en un embeddings y luego promediar las palabras de cada párrafo
    sentences_centroids_sentence = convert_to_vec(wordvectors, tokenized, mode = "sentence")  

    save_list("/home/klaus/discursos_politicos/data/original_sentences_parte", particion, original_text)
    save_list("/home/klaus/discursos_politicos/data/tokenization_parte", particion, tokenized)
    save_list("/home/klaus/discursos_politicos/data/centroids_sentence_parte", particion, sentences_centroids_sentence)
    del tokenized_text, tokenized, original_text 
    #pool.close()
    print("parte", particion)
    particion += 1

    
print("--- %s seconds ---" % (time.time() - start_time))




###################
# LIMPIAR MEMORIA #
###################

from IPython import get_ipython
get_ipython().magic('reset -sf')


