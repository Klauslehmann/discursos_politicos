import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

stop = stopwords.words('spanish')

df = pd.read_feather("data/score_filtered.feather")

# Palabras más cognitivas
df_cognitivas = df.sort_values("score")[0:5000]
df_cognitivas["text"] = df_cognitivas["text"].str.replace(r'\\n', '')


# Palabras más afectivas
df_afectivas = df.sort_values("score", ascending= False)[0:5000][["text", "score"]]
df_afectivas ["text"] = df_afectivas["text"].str.replace(r'\\n', '')


text_cog = " ".join(cat for cat in df_cognitivas.text)
text_afect = " ".join(cat for cat in df_afectivas.text)


# Frases cognitivas

wordcloud = WordCloud(width=1600, height=800, 
                      #background_color = "white", 
                      stopwords=stop,
                      max_words = 150).generate(text_cog )


plt.figure(figsize=(20,10) )
plt.imshow(wordcloud , interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)

plt.savefig('tesis/cuadros_tesis/wordcloud_most_cognitive_phrases.png', facecolor='k', bbox_inches='tight')
plt.show()



# Frases afectivas

wordcloud = WordCloud(width=1600, height=800, 
                      #background_color = "white", 
                      stopwords=stop,
                      max_words = 150).generate(text_afect)

plt.figure(figsize=(20,10) )
plt.imshow(wordcloud , interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)


plt.savefig('tesis/cuadros_tesis/wordcloud_most_afective_phrases.png', facecolor='k', bbox_inches='tight')
plt.show()

