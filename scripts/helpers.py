def pre_process_text(text):
  # Separar cada textos usando los puntos
  # Generar tokens
  # Sacar stopwords
  sentences = text.split(". ")
  token_sentences = [sentence.split() for sentence in sentences]
  tokens_filtered = [[ word for word in sentence if not word in stopwords.words('spanish')]  for sentence in token_sentences ]
  
  # Buscar etiqueta pos para cada una de las palabras de la lista affective 
  pos = []
  for word in tokens_filtered[0]:
    document = nlp(word)
    for token in document:
      pos.append((word, token.lemma_, token.pos_)) 
  
  # Dejar solo lo que sea sustantivo, adjetivo o verbo.
  important_words = [word[0] for word in pos if word[2] == "ADJ" or word[2] == "NOUN" or word[2] == "VERB" ]
  return important_words
