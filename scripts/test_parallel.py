import sys
sys.path.append('scripts/')

import time
import multiprocessing
import re
import string
from spacy.language import Language
import spacy
import pandas as pd
from gensim.models.fasttext import load_facebook_model
import copy
import numpy as np
import pickle
from helpers import convert_to_vec

nlp = spacy.load('es_core_news_md')

# Agregar una regla que separa usando el caracter \\n
@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc): 
    for token in doc[:-1]: 
        if token.text == "\\n": 
            doc[token.i + 1].is_sent_start = True
    return doc
nlp.add_pipe("set_custom_boundaries", first=True)


def pre_process_text(text, relevant_pos = ["NOUN", "ADJ", "VERB"], mode = "word", paragraph = False):
  
# =============================================================================
#   text= df.texto_dep[0]
#   relevant_pos = ["NOUN", "ADJ", "VERB"] # Separar cada texto usando los puntos
#   mode = "word"
#   paragraph = True
# =============================================================================
  
  if paragraph == True:
      punctuation = re.sub(r'\\|', '', string.punctuation)
      text_dep = text.translate(str.maketrans('', '', punctuation))
      text_dep = re.sub(r'\t',' ', text_dep) 
      text_dep= text_dep.replace('\\n', ' \\n ')      
      
  else: 
      # Remover puntuación y algunos caracteres molestos
      punctuation = re.sub(r'\\|\.', '', string.punctuation)
      text_dep = text.translate(str.maketrans('', '', punctuation))
      text_dep = re.sub(r'\\n','', text_dep).strip() 
      text_dep = re.sub(r'\t',' ', text_dep) 
      
      

  nlp_text = nlp(text_dep)
  #print("After:", [sent.text for sent in   nlp_text .sents])

  # Dejar solo las palabras importantes 
  if mode == "word":
    important_words = [
      [token.text for token in phrase if token.pos_ in relevant_pos and len(token.text) > 1 and token.text != "\\n" ] 
      for phrase in nlp_text.sents
  ]
  elif mode == "lemma":
    important_words = [
      [token.lemma_ for token in phrase if token.pos_ in relevant_pos and len(token.lemma_) > 1 and token.text != "\\n"] 
      for phrase in nlp_text.sents
  ]
  else:
    return ValueError("mode must be lemma or word")

  # Rescatar las oraciones iniciales 
  original_sentences = list(nlp_text.sents)
  original_sentences = list(map(lambda z:z.text, original_sentences))
  
  # Después del preprocesamiento quedan frases vacías. Para que todo funcione, es necesario hacer 
  # calzar las frases originales con las editadas  
  removed_indices = [counter for counter, elem in enumerate(important_words) if not elem ]
  for index in sorted(removed_indices, reverse=True):
    del original_sentences[index]
  
  important_words = [phrase for phrase in important_words if phrase]
  return important_words, original_sentences

def save_list(file_name, partition, list_object):
    file_name = "{file_name}{parte}".format(file_name = file_name, parte = partition)
    with open(file_name , "wb") as fp:
      pickle.dump(list_object, fp)



def store_file(fraction, particion):
    tokenized_text = [pre_process_text(t) for t in fraction]
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



# Cargar datos editados
df =  pd.read_feather("/home/klaus/discursos_politicos/data/edicion_inicial.feather")

# Cargar modelo de embeddings
wordvectors = load_facebook_model('/home/klaus/discursos_politicos/data/embeddings-m-model.bin') 


# Dividir texto en 4 partes
text_vector = copy.copy(df.texto_dep)
split_text = np.array_split(text_vector, 10)

    
#cpus = multiprocessing.cpu_count()


if __name__ == '__main__':
    start = time.perf_counter()

    # create processes
    processes = [multiprocessing.Process(target=store_file, args=[part, counter ]) 
                for counter, part in enumerate(split_text) ]

    # start the processes
    for process in processes:
        process.start()

    # wait for completion
    for process in processes:
        process.join()

    finish = time.perf_counter()

    print(f'It took {finish-start: .2f} second(s) to finish')


