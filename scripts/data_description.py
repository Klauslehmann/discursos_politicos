

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

################
# CARGAR DATOS #
################

# Cargar datos originales
df_filtrado =  pd.read_feather("data/score_filtered.feather")

# Cargar datos de nominate
nominate = pd.read_feather("/home/klaus/discursos_politicos/data/nominate_years.feather")
nominate = nominate.reset_index()

# Cargar listado de apellidos
alemanes = pd.read_feather("data/german_lastnames.feather")
italianos = pd.read_feather("data/italian_lastnames.feather")
ingleses = pd.read_feather("data/english_lastnames.feather")
franceses = pd.read_feather("data/french_lastnames.feather")
vascos = pd.read_feather("data/vascos_lastnames.feather")


###########################
# CREAR ALGUNAS VARIABLES #
###########################
df_filtrado["predictadura"] =  np.where(df_filtrado['anio'] <= 1973, 1, 0)
nominate['name'] = nominate ['name'].str.replace(r'...\d+', '').str.replace(r'_', ' ')

###########################
# ESTADÍSTICA DESCRIPTIVA #
###########################

# Generar histograma con los datos a nivel de frases
df_filtrado.score.describe()
n_bins = 150
plt.hist(df_filtrado[df_filtrado["score"] < 1.5].score, bins = n_bins)
#plt.savefig('scripts/reportes/histograma.png')
plt.figure().clear()

# Ver las frases más afectivas
afectivas = df_filtrado.sort_values(by=['score'], ascending=False)["text"][0:100]

# Ver las frases más cognitivas
cognitivas = df_filtrado.sort_values(by=['score'])["text"][0:100]

# Ver puntajes de ambos cosenos
plt.hist(df_filtrado.cos_affect, bins = n_bins)
plt.figure().clear()

plt.hist(df_filtrado.cos_cognitive, bins = n_bins)
plt.figure().clear()


# Ver puntaje de parlamentarios 
score_names = df_filtrado.groupby("nombre")["score"].mean().reset_index()

score_names .score.describe()
n_bins = 50
plt.hist(score_names.score, bins = n_bins)

score_names =  pd.merge(score_names, df_filtrado[ ["nombre", "polo"] ].groupby('nombre').first(), on='nombre')

most_affective = score_names[score_names["score"] >= 1.04]
most_affective.groupby("polo").size()

most_cognitive = score_names[score_names["score"] <= 1.0]
most_cognitive.groupby("polo").size()

# Ver relación entre sexo y emocionalidad
score_sex = df_filtrado.groupby("sexo")["score"].mean().reset_index()


# Ver relación entre edad y el indicador
bins= [25,35,45,55,65,75,85]
labels = ['25-34','35-44','45-54','55-64','65-74', "75-84"]
df_filtrado['edad_tramo'] = pd.cut(df_filtrado['edad_actual'], bins=bins, labels=labels, right=False)

bins2 = list(range(25, 90, 5))
labels2 = [ str(tramo) + "-" + str(tramo + 4)   for tramo in range(25, 85, 5)]
df_filtrado['edad_tramo2'] = pd.cut(df_filtrado['edad_actual'], bins=bins2, labels=labels2, right=False)


df_filtrado.edad_actual.describe()
score_age = df_filtrado[(df_filtrado["edad_actual"] < 100) & (df_filtrado["score"] > 0.76) ].groupby("edad_actual")["score"].mean().reset_index()

np.corrcoef(score_age.edad_actual, score_age.score)

score_age_cut = score_age[score_age.edad_actual <= 80]
plt.plot(score_age_cut ['edad_actual'], score_age_cut ['score'], color='red')

edad_tramo = df_filtrado.groupby("edad_tramo")["score"].mean().reset_index()
edad_tramo2 = df_filtrado.groupby("edad_tramo2")["score"].mean().reset_index()

plt.plot(score_age_cut ['edad_actual'], score_age_cut ['score'], color='red')
plt.plot(edad_tramo.edad_tramo, edad_tramo.score )
plt.plot(edad_tramo2.edad_tramo2, edad_tramo2.score )

# Ver puntaje de las tendencias politicas
score_tendency= df_filtrado.groupby("polo")["score"].mean()
score_tendency2= df_filtrado.groupby(["polo", "anio"])["score"].mean().reset_index()
score_tendency3= df_filtrado.groupby(["partido", "anio"])["score"].mean().reset_index()
score_tendency4= df_filtrado.dropna(subset = "polo").groupby("partido")["score"].mean().reset_index().sort_values("score")   


# Ver histograma para derecha e izquierda
izquierda =  df_filtrado[df_filtrado["polo"]  == "izquierda"]["score"]
centro =  df_filtrado[df_filtrado["polo"]  == "centro"]["score"]
derecha = df_filtrado[df_filtrado["polo"]  == "derecha"]["score"]


# =============================================================================
import seaborn as sns
sns.kdeplot(izquierda, bw=0.25)
sns.kdeplot(derecha, bw=0.25 )
sns.kdeplot(centro, bw=0.25 )

plt.show()


# =============================================================================


fig, ax = plt.subplots()
ax.hist(izquierda, bins = n_bins, alpha=0.5, label='izquierda')
ax.hist(derecha, bins = n_bins, alpha=0.5, label='derecha')
ax.hist(centro, bins = n_bins, alpha=0.5, label='centro')
ax.legend(loc='upper right')
ax.set_title('Simple plot')
fig

# Test kolmogorov-smirnoff
from scipy.stats import ks_2samp
import scipy.stats as stats

ks_2samp(izquierda, derecha)
ks_2samp(centro, derecha)

# Test de hipótesis
stats.ttest_ind(izquierda,
                derecha)

stats.ttest_ind(centro,
                derecha)

stats.ttest_ind(centro,
                izquierda)

stats.ttest_ind(df_filtrado[df_filtrado["sexo"] == "hombre"]["score"] ,
                df_filtrado[df_filtrado["sexo"] == "mujer"]["score"])

# gráfico por sexo
# kernel para distribuciones

###########################################
# REVISAR DATOS INDIVIDUALES DE POLÍTICOS #
###########################################

df_filtrado['lower_names'] = df_filtrado["nombre"].str.lower()

coloma_affective = df_filtrado[df_filtrado["lower_names"].str.contains("coloma")].sort_values("score", ascending = False)[["text", "score"]][0:10]
coloma_cognitive= df_filtrado[df_filtrado["lower_names"].str.contains("coloma")].sort_values("score")[["text", "score"]][0:10]
coloma_mean =  df_filtrado[df_filtrado["lower_names"].str.contains("coloma")]
np.mean(coloma_mean.score)

def get_most_affective(name, n = 10):
    affective = df_filtrado[df_filtrado["lower_names"].str.contains(name)].sort_values("score", ascending = False)[["text", "score"]][0:n]
    return affective 
    
def get_most_cognitive(name, n = 10):
    cognitive = df_filtrado[df_filtrado["lower_names"].str.contains(name)].sort_values("score")[["text", "score"]][0:n]
    return cognitive

def get_mean(name):
    filtered =  df_filtrado[df_filtrado["lower_names"].str.contains(name)]
    return np.mean(filtered.score)


get_mean("boric")    
get_mean("hertz")    
get_mean("moreira")    
get_mean("coloma")    


########################
# DESCRIPCIÓN TEMPORAL #
########################0

# Descripción de tendencias por polo-año
score_tendency_wide= score_tendency2[score_tendency2["anio"] < 2022].pivot(index='anio', columns='polo', values='score')

# Moving average con 3 rezagos
rolling = score_tendency_wide.rolling(window=3)
rolling_mean = rolling.mean()

# Graficar las 3 tendencias a lo largo del tiempo
fig, ax = plt.subplots()
rolling_mean.loc[:1973].plot(ax = ax, color = ["yellow", "green", "red"])
rolling_mean.loc[1990:2021].plot(ax = ax, color = ["yellow", "green", "red"])

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

legend_without_duplicate_labels(ax)
fig

# Mirar lo que pasa el 2003 con la izquierda
izquierda_2003 = df_filtrado[(df_filtrado["anio"] == 2003) & (df_filtrado["polo"]=="izquierda") ]
left_stats = df_filtrado[df_filtrado["polo"] == "izquierda" ].groupby("anio").agg({"score": ["mean", "median", "min", "max", "std"]})

# Revisar trayectorias temporales para parlamentarios que tengan varios años
largo = df_filtrado[df_filtrado["anio"] >= 1990] .groupby(["nombre", "anio"]).first().reset_index()[["nombre", "anio"]]
vinagres = largo .groupby("nombre").count()
vinagres = vinagres[vinagres ["anio"] > 20]

larga_duracion = pd.merge(df_filtrado, vinagres, on = "nombre", how = "right")
media_anio = larga_duracion.groupby([ "edad_tramo2","nombre"]).agg(conteo =  ("score", "size"),
                                                                   media = ("score", "mean")
                                                                   ).reset_index()

moreira = media_anio[media_anio["nombre"] == "Iván Moreira Barros"]
escalona = media_anio[media_anio["nombre"] == "Camilo Escalona Medina"]


fig, axs = plt.subplots(8, 3)
plt.subplots_adjust(hspace=0.5)
fig.suptitle("Daily closing prices", fontsize=18, y=0.95)

for name, ax in zip(pd.unique(media_anio .nombre), axs.ravel()):
    media_anio[media_anio["nombre"] == name][["edad_tramo2", "media"] ].set_index("edad_tramo2").plot(ax=ax)
    
    # chart formatting
    #ax.set_title(name)
    ax.get_legend().remove()
    ax.set_xlabel("")

plt.show()


########################
# Trabajar con tópicos #
########################

df_test =  pd.read_feather("data/topics_test_paragraph.feather")

# Explorar cantidad de tópicos
df_test.groupby("pole").size()
df_test .groupby("topic").size()  

# Puntaje de cada tópico
score_topic =  df_test[df_test["score_topic"] >= 0.7].groupby("topic")["score"].mean().reset_index()

x = list(range(0, 11))

fig, ax = plt.subplots()
ax.scatter(x, score_topic.score)
ax.set_title('Simple plot')
ax.set_xticks(x)
ax.set_xticklabels(score_topic.topic, rotation=45)

fig
fig.show()

# Revisar puntaje de cada tópico por cada polo político
score_topic_pole =  df_test[df_test["score_topic"] >= 0.8].groupby(["topic", "pole"])["score"].mean().reset_index()
izquierda = score_topic_pole [ (score_topic_pole ["pole"] == "izquierda")]
derecha = score_topic_pole [ (score_topic_pole ["pole"] == "derecha")]

fig, ax = plt.subplots()
ax.scatter(x, izquierda.score)
ax.scatter(x, derecha.score )
ax.legend(["izquierda" , "derecha"])
ax.set_title('Emotividad en cada tópico, según polaridad política')
ax.set_xticks(x)
ax.set_xticklabels(score_topic.topic, rotation=45)
fig
fig.show()


# Revisar emocionalidad en cada uno de los tópicos
emo_left =  df_test[df_test["pole"]== "izquierda"].groupby([ "topic"])["emotion"].mean().reset_index()
emo_right=  df_test[df_test["pole"]== "derecha"].groupby([ "topic"])["emotion"].mean().reset_index()

emo_right.emotion / emo_left.emotion  

# Revisar tópicos a lo largo del tiempo
topic_year =  df_test.groupby([ "topic", "year"])["score"].mean().reset_index()

score_topic_wide= topic_year [topic_year ["year"] < 2022].pivot(index='year', columns='topic', values='score')
score_topic_wide = score_topic_wide[["cultura", "impuestos", "educación"]]

# Moving average con 3 rezagos
rolling = score_topic_wide .rolling(window=3)
rolling_mean = rolling.mean()


fig, ax = plt.subplots()
rolling_mean  .loc[:1973].plot(ax = ax, color = ["yellow", "green", "red"])
rolling_mean .loc[1990:2021].plot(ax = ax, color = ["yellow", "green", "red"])


legend_without_duplicate_labels(ax)
fig
fig.show()

# Revisar tópicos
topico = "educación"
frases= df_test.sort_values(by = ["score_topic"], ascending = False)[df_test["topic"]==topico]


#############
# Regresión #
#############

import statsmodels.formula.api as smf

df_test.columns
df_filtrado.columns

# Add name 
df_topics = pd.merge(df_test, df_filtrado[["id_phrase", "nombre"]], on='id_phrase')
df_topics["name"] = df_topics["name"].str.lower()
df_topics = df_topics.dropna()


# Editar el nombre que viene en la tabla nominate, para asegurar el match con la tabla que contiene los textos
nominate["name"] = nominate["name"].str.strip()
nominate["name"][nominate["name"] == "adriana muñoz d'albora"] = "adriana muñoz d' albora"
nominate["name"][nominate["name"] == "alberto cardemil herrera"] = "alberto eugenio cardemil herrera"
nominate["name"][nominate["name"] == "alejandro sule fernández"] = "alejandro miguel sule fernández"
nominate["name"][nominate["name"] == "andrés egaña respaldiza"] = "andrés antonio egaña respaldiza"
nominate["name"][nominate["name"] == "carlos vilches guzmán"] = "carlos alfredo vilches guzmán"
nominate["name"][nominate["name"] == "cristián leay morán"] = "cristian antonio leay morán"
nominate["name"][nominate["name"] == "eduardo cerda garcía"] = "eduardo antonio cerda garcía"
nominate["name"][nominate["name"] == "renán fuentealba vildósola"] = "francisco renán fuentealba vildósola"
nominate["name"][nominate["name"] == "gastón von mühlenbrock zamora"] = "gastón von muhlenbrock zamora"
nominate["name"][nominate["name"] == "giovanni calderón bassi"] = "giovanni oscar calderón bassi"
nominate["name"][nominate["name"] == "raúl sunico galdames"] = "raúl súnico galdames"
nominate["name"][nominate["name"] == "ramón pérez opazo"] = "ramón segundo pérez opazo"
nominate["name"][nominate["name"] == "pedro muñoz aburto"] = "pedro héctor muñoz aburto"
nominate["name"][nominate["name"] == "juan bustos ramírez"] = "juan josé bustos ramírez"
nominate["name"][nominate["name"] == "pedro pablo álvarez salamanca büchi"] = "pedro pablo alvarez-salamanca büchi"

# Editar algunos nombres de la tabla de la regresión, para asegurar el match
df_topics["name"][df_topics["name"] == "guido girardi brière"] = "guido girardi briere"
df_topics["name"][df_topics["name"] == "jose antonio galilea vidaurre"] = "josé antonio galilea vidaurre"


# Add nominate score
nominate_year = nominate.groupby(["name", "year"])["coord1D"].mean().reset_index()
df_topics ["name"] = df_topics ["name"].str.strip()
df_topics ["year"] = df_topics ["year"].astype(int)
df_topics = pd.merge(df_topics.reset_index(), nominate_year, on = ["name", "year"], how = "left")
df_topics["nominate"] = abs(df_topics["coord1D"])


# Hacer algunas comprobaciones
df_topics[(df_topics.coord1D.isnull()) & (df_topics.year >= 2002) ].shape
df_topics.dropna()[df_topics.year >= 2002].shape
df_topics[df_topics.year >= 2002].shape
not_matched = df_topics[(df_topics.coord1D.isnull()) & (df_topics.year >= 2002) ].groupby(["name", "year"], as_index = False).first()[["name", "year"]]
nominate_year[nominate_year['name'].str.contains('sule')].groupby(["name", "year"] , as_index = False).first()[["name", "year", "coord1D"]]


# Create variables
bins= [25,35,45,55,65,75,85] 
labels = ['25-34','35-44','45-54','55-64','65-74', "75-84"]
df_topics['age_group'] = pd.cut(df_topics['age'], bins=bins, labels=labels, right=False)
df_topics["age2"] = list(map(lambda x:x*x, df_topics["age"] ))


df_topics["w2"] = list(map(lambda x:x*x, df_topics["nominate"] ))
df_standard = df_topics[["w2", "score", "nominate"]]
df_topics[['w2_standard', 'score_standard', "nominate_standard"]] = (df_standard-df_standard.mean())/df_standard.std()


# Create last name origin. The first step is look for a coincidence with any of the lists.
# Second step: use the API to get the rest of the lastnames

def find_lastname(name, lastname_list): 
    for n in name.lower().split()[-2:]:
        edit_n = re.sub("á", "a", n)
        edit_n = re.sub("é", "e", edit_n )
        edit_n = re.sub("í", "i", edit_n )
        edit_n = re.sub("ó", "o", edit_n )
        edit_n = re.sub("ú", "u", edit_n )
        if edit_n in lastname_list:
            return 1
    return 0


lastnames = df_topics[["name"]].groupby("name").first().reset_index()
lastnames["vasco"] = [find_lastname(n, vascos.lastname2.values) for n in lastnames.name]
lastnames["aleman"] = [find_lastname(n, alemanes.lastname2.values) for n in lastnames.name]
lastnames["ingles"] = [find_lastname(n, ingleses.lastname2.values) for n in lastnames.name]
lastnames["italiano"] = [find_lastname(n, italianos.lastname2.values) for n in lastnames.name]

df_topics = pd.merge(df_topics, lastnames, on='name')



    


##### Without nominate

df_topics_no_nominate = df_topics.loc[:, ~df_topics.columns.isin(['nominate', "nominate_standard", 'coord1D', "w2", "w2_standard"])].dropna() 

# =============================================================================
# model = smf.ols(formula="score_standard ~  C(age_group) + C(sex) + C(year) + C(topic, Treatment(reference='educación')) + C(pole, Treatment(reference='derecha')) + vasco + aleman + ingles",                 
#                 data=df_topics_no_nominate   ).fit()
# print(model.summary())
# =============================================================================

# full
model_cluster = smf.ols(formula="score_standard ~  C(age_group) + C(sex) + C(year) + C(topic, Treatment(reference='educación')) + C(pole, Treatment(reference='derecha')) + vasco + aleman + ingles",   #                
                data=df_topics_no_nominate   ).fit(cov_type='cluster',
                    cov_kwds={'groups': df_topics_no_nominate   ['name']},
                    use_t=True)
print(model_cluster.summary())

# no fixed year
model_cluster = smf.ols(formula="score_standard ~  C(age_group)   + C(sex)  + C(topic, Treatment(reference='educación')) + C(pole, Treatment(reference='derecha')) + vasco + aleman + ingles",                 
                data=df_topics_no_nominate   ).fit(cov_type='cluster',
                    cov_kwds={'groups': df_topics_no_nominate   ['name']},
                    use_t=True)

print(model_cluster.summary())
                            
# no fixed topic
model_cluster = smf.ols(formula="score_standard ~  C(age_group)  + C(sex) + C(year)  + C(pole, Treatment(reference='derecha')) + vasco + aleman + ingles",   #                
                data=df_topics_no_nominate   ).fit(cov_type='cluster',
                    cov_kwds={'groups': df_topics_no_nominate   ['name']},
                    use_t=True)
print(model_cluster.summary())



##### Including nominate


df_topics_nominate = df_topics.dropna() 

# =============================================================================
# model_nominate = smf.ols(formula="score_standard ~  C(age_group) + C(sex) + C(year) + C(topic, Treatment(reference='educación')) + C(pole, Treatment(reference='derecha')) + vasco + aleman + ingles",                 
#                 data=df_topics_nominate ).fit()
# print(model_nominate .summary())
# 
# =============================================================================


# agruár edad en tres
# graficar edad en el tiempo
# agregr regresor senador 
# paso intermedio: achicar la muestra con los mismos regresores

# full
model_cluster_nominate = smf.ols(formula="score_standard ~ w2_standard + C(age_group) + C(sex) + C(year) + C(topic, Treatment(reference='educación')) + C(pole, Treatment(reference='derecha')) + vasco + aleman + ingles",                 
                data=df_topics_nominate ).fit(cov_type='cluster',
                    cov_kwds={'groups': df_topics_nominate ['name']},
                    use_t=True)
print(model_cluster_nominate.summary())

# no fixed year
model_cluster_nominate = smf.ols(formula="score_standard ~ w2_standard + C(age_group) + C(sex)  + C(topic, Treatment(reference='educación')) + C(pole, Treatment(reference='derecha')) + vasco + aleman + ingles",                 
                data=df_topics_nominate ).fit(cov_type='cluster',
                    cov_kwds={'groups': df_topics_nominate ['name']},
                    use_t=True)
print(model_cluster_nominate.summary())

# no fixed topic
model_cluster_nominate = smf.ols(formula="score_standard ~ w2_standard + C(age_group) + C(sex) + C(year)  + C(pole, Treatment(reference='derecha')) + vasco + aleman + ingles",                 
                data=df_topics_nominate ).fit(cov_type='cluster',
                    cov_kwds={'groups': df_topics_nominate ['name']},
                    use_t=True)
print(model_cluster_nominate.summary())




