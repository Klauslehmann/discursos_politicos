

from gensim.models.keyedvectors import KeyedVectors
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib as plt

cantidad = 30000
wordvectors_file_vec = "/home/klaus/discursos_politicos/data/embeddings-m-model.vec"
wordvectors = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)


# Ejemplo con colores
colores =  wordvectors.most_similar(positive=['rojo'],  topn = 5)
colores_table = pd.DataFrame(colores, columns= ["palabra", "similitud"])
colores_table.to_csv("tesis/cuadros_tesis/ejemplo_embeddings_colores.csv")


# Ejemplo con analogía rey y reina
most_similars =  wordvectors.most_similar(positive=['rey','mujer'], negative=['hombre'], topn = 5)
analogia = pd.DataFrame(most_similars , columns= ["palabra", "similitud"])
analogia.to_csv("tesis/cuadros_tesis/ejemplo_embeddings_analogia.csv")


# Generar datos para herramienta de visualización de tensorflow
vectores =  wordvectors.vectors 
vectores_df = pd.DataFrame(vectores )
words = wordvectors.index_to_key
words_df = {"word": words}
words_df = pd.DataFrame(words_df )

vectores_df .to_csv('tesis/cuadros_tesis/vectores.tsv', sep="\t", index=False)
words_df  .to_csv('tesis/cuadros_tesis/vectores_metadata.tsv', sep="\t", index=False)

##########################
# Ejemplo gráfico con pca
##########################

# Seleccionar palabras para el ejemplo
extra_words = ["elefante", "león", "pan", "jamón", "cerveza", "vino", "casa", "auto"]

example_words = ["perro", "conejo", "caballo", "mono", 
                 "verde", "rojo", "azul", "amarillo", 
                 "pizza", "hamburguesa", "arroz"
                 ]

example_words.extend(extra_words) 

# Generar la matriz con los vectores de palabras para el ejemplo 
word_vectors = np.array([wordvectors[w] for w in example_words ])

# Obtener la varianza explicada de las 2 primeras componentes
pca_function = PCA(n_components=2)
principalComponents = pca_function.fit_transform(word_vectors)
variance =  pca_function.explained_variance_ratio_
 
varianza_pca = pd.DataFrame({"varianza": variance, "componente": ["componente1", "componente2"] } )

# Extraer las 2 primeras componentes para hacer el gráfico
pca = PCA().fit_transform(word_vectors)
twodim = pca[:,:2]
pca_example = pd.DataFrame({
    "word": example_words,
    "d1": twodim[:,0], 
    "d2": twodim[:,1] })

# Graficar
fig, ax = plt.subplots()
x = pca_example.d1
y = pca_example.d2

ax.scatter(x, y)

for i, txt in enumerate(example_words):
    ax.annotate(txt, (x[i], y[i]))

fig

pca_example.to_csv('tesis/cuadros_tesis/pca_example.csv')
varianza_pca.to_csv('tesis/cuadros_tesis/pca_retained_variance.csv')

# No utilizado
# =============================================================================
# rey =  wordvectors["rey"]  
# mujer =  wordvectors["mujer"]  
# hombre = wordvectors["hombre"]
# 
# candidato = (rey  + mujer) - hombre
# 
# words = wordvectors.most_similar(candidato)
# result =  [w[0] for w in words if w[0] not in ["rey", "mujer", "hombre"]][0:3]
# 
# print(result)
# =============================================================================
