from nltk.corpus import stopwords
import spacy
import re
import string
import numpy as np
nlp = spacy.load('es_core_news_md')

def pre_process_text(text):
  #text = reducida.texto_dep[13]
  # Separar cada texto usando los puntos
  sentences = text.strip().rstrip('.').split(". ")
  
  # Generar tokens
  token_sentences = [sentence.split() for sentence in sentences]
  
  # Remover puntuación y algunos caracteres molestos
  punctuation = re.sub(r'\\', '', string.punctuation)
  token_sentences = [[word.translate(str.maketrans('', '', punctuation)) for word in sentence] 
  for sentence in token_sentences]
  token_sentences = [[re.sub(r'\\n|,','', word) for word in sentence] for sentence in token_sentences]
  
  # Sacar stopwords y eliminar caracteres en blanco
  text_tokenized = [[ re.sub(r'\\n|,','', word)  for word in sentence if not word in stopwords.words('spanish')]  for sentence in token_sentences ]
  text_tokenized = [[word for word in sentence if word != '' ]   for sentence in text_tokenized ]
  text_tokenized = [sentence for sentence in text_tokenized if sentence]
  
  # Función para hacer POS
  def get_pos(word):
    document = nlp(word)
    for token in document:
      pos = (word, token.lemma_, token.pos_)
    return pos
  
  # Dejamos solo el stemming de las palabras relevantes
  pos_phrases =  [ [get_pos(word)  for word in phrase ] for phrase in text_tokenized]
  important_words = [[word[1] for word in phrase if word[2] == "ADJ" or word[2] == "NOUN" or word[2] == "VERB"] for phrase in pos_phrases ]
  important_words = [x for x in important_words if x]
  
  # Sacar las palabras que tienen solo un caracter. Lo más probable es que sean enumeraciones de un parlamentario, al momento de hablar
  important_words = [[word for word in phrase if len(word) > 1] for phrase in important_words]
  return important_words



# Función que calcula el centroide de un conjunto de vectores
def get_centroid(vectors):
  vectors_array = np.asarray(vectors)
  centroid = np.mean(vectors_array, axis=0)
  return centroid

