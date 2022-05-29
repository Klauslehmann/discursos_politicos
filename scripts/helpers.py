from nltk.corpus import stopwords
import spacy
import re
import string
import numpy as np
from numpy import dot
from numpy.linalg import norm

nlp = spacy.load('es_core_news_md')


# Función que calcula el centroide de un conjunto de vectores
def get_centroid(vectors):
  vectors_array = np.asarray(vectors)
  centroid = np.mean(vectors_array, axis=0)
  return centroid



# Función para preprocesar el texto
def pre_process_text(text, relevant_pos = ["NOUN", "ADJ", "VERB"]):
  
  text = df.texto_dep[0]
  relevant_pos = ["NOUN", "ADJ", "VERB"]
  #Separar cada texto usando los puntos
  
  # Remover puntuación y algunos caracteres molestos
  punctuation = re.sub(r'\\|\.', '', string.punctuation)
  text_dep = text.translate(str.maketrans('', '', punctuation))
  text_dep = re.sub(r'\\n','', text_dep).strip() 
  text_dep = re.sub(r'\t',' ', text_dep) 
  
  # Pasar por NLP
  nlp_text = nlp(text_dep)
  
  # Dejar solo las palabras importantes 
  important_words = [
      [token.lemma_ for token in phrase if token.pos_ in relevant_pos and len(token.lemma_) > 1] 
      for phrase in nlp_text.sents
  ]
  
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


# Calcular la distacia coseno entre dos vectores
def get_cosine(vector1, vector2):
  result = dot(vector1, vector2)/(norm(vector1)*norm(vector2))
  return result


