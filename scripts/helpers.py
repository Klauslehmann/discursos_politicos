from nltk.corpus import stopwords
import spacy
import re
nlp = spacy.load('es_core_news_md')

def pre_process_text(text):
  # Separar cada texto usando los puntos
  # Generar tokens
  # Sacar stopwords
  sentences = text.split(". ")
  token_sentences = [sentence.split() for sentence in sentences]
  text_tokenized = [[ re.sub(',','', word)  for word in sentence if not word in stopwords.words('spanish')]  for sentence in token_sentences ]
  
  def get_pos(word):
    document = nlp(word)
    for token in document:
      pos = (word, token.lemma_, token.pos_)
    return pos
  
  pos_phrases =  [ [get_pos(word)  for word in phrase ] for phrase in text_tokenized]
  important_words = [[word[1] for word in phrase if word[2] == "ADJ" or word[2] == "NOUN" or word[2] == "VERB"] for phrase in pos_phrases ]
  return important_words


#pre_process_text(reducida.texto_dep[0])








