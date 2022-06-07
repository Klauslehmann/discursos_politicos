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
def pre_process_text(text, relevant_pos = ["NOUN", "ADJ", "VERB"], mode = "word"):
  
  # text = df.texto_dep[0]
  # relevant_pos = ["NOUN", "ADJ", "VERB"]
  #Separar cada texto usando los puntos
  
  # Remover puntuación y algunos caracteres molestos
  punctuation = re.sub(r'\\|\.', '', string.punctuation)
  text_dep = text.translate(str.maketrans('', '', punctuation))
  text_dep = re.sub(r'\\n','', text_dep).strip() 
  text_dep = re.sub(r'\t',' ', text_dep) 
  
  # Pasar por NLP
  nlp_text = nlp(text_dep)
  
  # Dejar solo las palabras importantes 
  if mode == "word":
    important_words = [
      [token.text for token in phrase if token.pos_ in relevant_pos and len(token.text) > 1] 
      for phrase in nlp_text.sents
  ]
  elif mode == "lemma":
    important_words = [
      [token.lemma_ for token in phrase if token.pos_ in relevant_pos and len(token.lemma_) > 1] 
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


# Calcular la distacia coseno entre dos vectores
def get_cosine(vector1, vector2):
  result = dot(vector1, vector2)/(norm(vector1)*norm(vector2))
  return result

# Sacar palabras que no interesan
def remove_unimportant_words(text, relevant_pos = ["NOUN", "ADJ", "VERB"], mode = "word"):
  vocab = []
  seen = set()
  spacy = nlp(text)
  
  if mode == "lemma":
    for word in spacy:
      if word.lemma_ not in seen and not word.is_punct and not word.is_digit and not word.is_stop and len(word) > 1 and word.pos_ in relevant_pos:
        vocab.append(word.lemma_)
      seen.add(word.lemma_)
  elif mode == "word":
    for word in spacy:
      if word.lemma_ not in seen and not word.is_punct and not word.is_digit and not word.is_stop and len(word) > 1 and word.pos_ in relevant_pos:
        vocab.append(word.text)
      seen.add(word.text)
  else:
    return ValueError("mode must be lemma or word")
  return vocab

# Dejar solo lo que sea sustantivo, adjetivo o verbo
def filter_important_words(words_list, mode = "word", relevant_pos = ["NOUN", "ADJ", "VERB"]):
  filtered_words = []
  for word in words_list:
    document = nlp(word)
    for token in document:
      if token.pos_ in relevant_pos:
        filtered_words.append((word, token.lemma_, token.pos_)) 
  if mode == "word":
    filtered_words = [word[0] for word in filtered_words]
  elif mode == "lemma":
    filtered_words = [word[1] for word in filtered_words]
  return filtered_words
