###################################################3

# Calcular la distacia coseno
def get_cosine(vector1, vector2):
  result = dot(vector1, vector2)/(norm(vector1)*norm(vector2))
  return(result)


z = 1.1
drop_rows = round(df_affective.shape[0] / z) 
df_affective_final = df_affective[drop_rows:df_affective.shape[0]]

drop_rows = round(df_cognitive.shape[0] / z) 
df_cognitive_final = df_cognitive[drop_rows:df_cognitive.shape[0]]

affective_vectors_list = [wordvectors.wv[word] for word in df_affective_final.word]
affective_vectors_array = np.asarray(affective_vectors_list)
affective_vector = np.mean(affective_vectors_array, axis=0)

cognitive_vectors_list = [wordvectors.wv[word] for word in df_cognitive_final.word]
cognitive_vectors_array = np.asarray(cognitive_vectors_list)
cognitive_vector = np.mean(cognitive_vectors_array, axis=0)

get_cosine(affective_vector, cognitive_vector)

ejemplo = [[["consciente", "conocimiento", "consulté", "vigilar"], ["llanto", "alegría", "odio", "amor"]] ]
text_vectors = [[wordvectors.wv[word] for word in sentence ]  for sentence in ejemplo ]
sentences_centroids = [[get_centroid(sentence) for sentence in text ] for text  in text_vectors]
A = get_cosine(sentences_centroids[0][1], affective_vector)
C = get_cosine(sentences_centroids[0][1], cognitive_vector)

polarity =  (A ) / (C  )
print(polarity)

###################################################3

