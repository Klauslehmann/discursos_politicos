library(quanteda)
library(feather)
library(tidyverse)
library(tictoc)
library(quanteda.textmodels)
library(quanteda.textstats)
library(quanteda.textplots)

# Cargar fechas relevantes en la línea de tiempo 
dates <- read_feather("tesis/cuadros_tesis/fechas_peaks_camaras.feather")

# Cargar datos de discursos políticos
editado <- arrow::read_feather("data/score_filtered.feather")
polos <-  arrow::read_feather("tesis/cuadros_tesis/polos_df.feather")

stopwords <- c(quanteda::stopwords("es"), "señor", "señores", "proyecto", "presidente", "presidenta", "dicho", "honorable", "don",
               "nyo", "garcia", "sanfuentes", "bulnes", "ibañez", "arnello", "ochagavia", "phillips", 
               "von", "muhlenbrock", "gustavo", "mühlenbrock", "nel", "solo", "sólo", "n°", "vieragallo", "sanfuentes"
)

# Tratamiento a la tabla que contiene los peaks afectivos
dates <- dates %>% 
  mutate(month = str_sub(mes, 1, 7))
  


# Tratamiento a la tabla que contiene los discursos parlamentarios
editado2 <- editado %>%
  mutate(polo = if_else(partido == "chileprimero", "derecha", polo)) %>% 
  filter(!is.na(polo) & polo != "centro") %>% 
  sample_frac(0.3) %>%
  mutate(setentas = if_else(anio <= 1973, 1, 0)) %>%
  mutate(month = format(as.Date(fecha, format="%Y-%m-%d"),"%Y-%m")) %>% 
  select(text = text_edit, anio, month,  polo, partido, setentas, nombre, n_words, role, score)  %>% 
  mutate(polo_fct = as.numeric(as.factor(polo)) - 1)



# Convertir intervenciones a matriz dfm
dfm <- editado2  %>% 
  corpus() %>% 
  tokens(remove_punct = T, remove_symbols = T) %>%
  tokens_remove(stopwords) %>% 
  tokens_ngrams( n = 1:2) %>% 
  tokens_select(pattern = stopwords, selection = "remove") %>%
  tokens_select(min_nchar = 2, selection = "remove") %>% 
  dfm() 


correlacion <- dfm  
  #dfm_subset(subset = anio >= 1990 & anio  <= 2022) %>% 
  #dfm_subset(subset = role == "diputado")

correlacion <- textstat_keyness(correlacion, target = correlacion$month == "1973-09")

textplot_keyness(correlacion)



create_keyness <- function(dfm, start, end, target) {
  new_dfm <- dfm %>% 
    dfm_subset(subset = anio >= start & anio  <= end) %>% 
    dfm_subset(subset = polo != "centro") 
  
  correlacion <- textstat_keyness(new_dfm, target = new_dfm$polo == target)
  
  textplot_keyness(correlacion)
  
  return(correlacion)
}


setentas1 <- create_keyness(dfm, 1965, 1970, "izquierda")
setentas2 <- create_keyness(dfm, 1971, 1973, "izquierda")
noventas_1 <- create_keyness(dfm, 1990, 1994, "izquierda")
noventas_2 <- create_keyness(dfm, 1995, 1999, "izquierda")
dosmil_1 <- create_keyness(dfm, 2000, 2004, "izquierda")
dosmil_2 <- create_keyness(dfm, 2005, 2009, "izquierda")
dosmil_3 <- create_keyness(dfm, 2010, 2014, "izquierda")
dosmil_4 <- create_keyness(dfm, 2015, 2020, "izquierda")

periodos <- list(setentas1, setentas2,noventas_1, noventas_2, dosmil_1, dosmil_2, dosmil_3, dosmil_4 )




tables_head <- map(periodos, ~.x %>% dplyr::slice(1:500) ) 

names(tables_head) <- c("setentas1", "setentas2", "noventas_1", "noventas_2", "dosmil_1", "dosmil_2", "dosmil_3", "dosmil_4") 
tables_head2 <- imap(tables_head, ~as.data.frame(.x)  %>% 
       mutate(anio = .y)
       ) %>% 
  bind_rows()


tables_head_derecha <- map(periodos, ~.x %>% as.data.frame() %>%   arrange( chi2) %>%   dplyr::slice(1:500) %>% dplyr::mutate(chi2 = abs(chi2) ) ) 
names(tables_head_derecha) <- c("setentas1", "setentas2", "noventas_1", "noventas_2", "dosmil_1", "dosmil_2", "dosmil_3", "dosmil_4") 

tables_head_derecha$setentas1 %>% view()
library(wordcloud)
library(wordcloud2)
library(RColorBrewer)
library(htmlwidgets) 
#########################
# WORDCLOUD IZQUIERDA
#########################

# Izquierda antes del golpe
set.seed(1234) # for reproducibility 
df <- data.frame(word = tables_head$setentas1$feature,
                 freq = tables_head$setentas1$chi2)
hw <- wordcloud2(data=df, size = 0.7, shape = 'pentagon')
saveWidget(hw,"tesis/cuadros_tesis/left_seventies.html",selfcontained = F)

# Izquierda en el último periodo
set.seed(1234) # for reproducibility 
df <- data.frame(word = tables_head$dosmil_4$feature, freq = tables_head$dosmil_4$chi2)
hw <- wordcloud2(data=df, size = 0.7, shape = 'pentagon')
saveWidget(hw,"tesis/cuadros_tesis/left_last_time.html",selfcontained = F)

#####################
# WORDCLOUD DERECHA #
#####################

set.seed(1234) # for reproducibility 
df <- data.frame(word = tables_head_derecha$setentas1$feature,
                 freq = tables_head_derecha$setentas1$chi2)
hw <- wordcloud2(data=df, size = 0.7, shape = 'pentagon')
saveWidget(hw,"tesis/cuadros_tesis/right_seventies.html",selfcontained = F)


set.seed(1234) # for reproducibility 
df <- data.frame(word = tables_head_derecha$noventas_1$feature,
                 freq = tables_head_derecha$noventas_1$chi2)
hw <- wordcloud2(data=df, size = 0.7, shape = 'pentagon')
saveWidget(hw,"tesis/cuadros_tesis/right_seventies.html",selfcontained = F)



tables_head2 %>% 
  mutate(anio = fct_relevel(as.factor(anio), c("setentas1", "setentas2", "noventas_1", "noventas_2", "dosmil_1", "dosmil_2", "dosmil_3", "dosmil_4"))) %>% 
  ggplot(aes(x = reorder(feature, desc(chi2))  , y = chi2 )) +
  geom_bar(stat = "identity") +
  coord_flip() +
  facet_wrap(~anio, ncol = 4)

library(patchwork)

noventas_1 %>% 
  dplyr::slice(1:10) %>% 
  inner_join(noventas_2 %>% 
              dplyr::slice(1:10) %>%
              as.data.frame() %>%  
              dplyr::select(feature),
            by = "feature")


plot_setentas1 <- textplot_keyness(setentas1) 
plot_setentas2 <- textplot_keyness(setentas2) 
plot_noventas_1 <-textplot_keyness(noventas_1)
plot_noventas_2 <-textplot_keyness(noventas_2)
plot_dosmil_1 <- textplot_keyness(dosmil_1)
plot_dosmil_2 <- textplot_keyness(dosmil_2)
plot_dosmil_3 <- textplot_keyness(dosmil_3)
plot_dosmil_4 <- textplot_keyness(dosmil_4)


############################
# Graficar algunas palabras
############################


left_words_setentas <- c("imperialismo", "trabajadores", "pueblo", "lucha", "huelga")
left_words_dosmil <- c("trabajadoras", "trabajadores", "derechos", "mujeres", "dictadura")
left_words_noventas <- c("derecha", "democracia", "dictadura", "derechos_humanos", "neoliberal", "trabajadores_trabajadoras")
right_words <- c("marxista", "mercado", "agricultores", "agrícola", "agricultura", "predio", "propiedad", "delincuencia", "impuestos")
words <- c(left_words_setentas, left_words_dosmil, left_words_noventas)


izquierda <- dfm %>% 
  dfm_subset( polo == "izquierda") %>% 
  dfm_select( words) %>% 
  textstat_frequency(groups = anio) %>% 
  as.data.frame()

derecha <- dfm %>% 
  dfm_subset(polo == "derecha") %>% 
  dfm_select(right_words) %>% 
  textstat_frequency(groups = anio) %>% 
  as.data.frame()


# Palabras para la derecha

create_plot_word(derecha, "mercado")
create_plot_word(derecha, "agricultores")
create_plot_word(derecha, "agricultura")
create_plot_word(derecha, "agrícola")
create_plot_word(derecha, "predio")
create_plot_word(derecha, "propiedad")
create_plot_word(derecha, "impuestos")
create_plot_word(derecha, "delincuencia")



# Palabras para la izquierda

create_plot_word(izquierda, "trabajadores")
create_plot_word(izquierda, "pueblo")
create_plot_word(derecha, "democracia")
create_plot_word(izquierda, "trabajadoras")
create_plot_word(izquierda, "mujeres")
create_plot_word(izquierda, "neoliberal")
create_plot_word(izquierda, "derechos_humanos")
create_plot_word(izquierda, "trabajadores_trabajadoras")




weights <- editado2 %>% 
  filter(polo == "izquierda") %>% 
  group_by(anio) %>% 
  summarise(total_words = sum(n_words)) %>% 
  ungroup() %>% 
  mutate(total_words = total_words / 100,
         anio = as.character(anio)
         ) 


anios <- data.frame(anio = as.character(1965:2021), frequency0 = 0)

izquierda %>% 
  rename(anio = group) %>% 
  left_join(weights, by = c( "anio")  ) %>% 
  mutate(frequency = frequency / total_words) %>% 
  bind_rows(data.frame(anio = as.character(1974:1989), frequency = NA)) %>% 
  filter(feature == "dictadura" | is.na(feature)) %>% 
  full_join(anios, by = "anio") %>% 
  mutate(frequency = if_else(is.na(total_words) & (anio <= 1973 | anio >= 1990) , 0, frequency)) %>% 
  ggplot(aes(x = anio, y = frequency, group = 1)) +
  geom_line() +
  theme(axis.text.x = element_text(angle = 60))

create_plot_word  <- function(data, word) {
  data %>% 
    rename(anio = group) %>% 
    left_join(weights, by = c( "anio")  ) %>% 
    mutate(frequency = frequency / total_words) %>% 
    bind_rows(data.frame(anio = as.character(1974:1989), frequency = NA)) %>% 
    filter(feature == word | is.na(feature)) %>% 
    full_join(anios, by = "anio") %>% 
    mutate(frequency = if_else(is.na(total_words) & (anio <= 1973 | anio >= 1990) , 0, frequency)) %>% 
    ggplot(aes(x = anio, y = frequency, group = 1)) +
    geom_line() +
    theme(axis.text.x = element_text(angle = 60))
  
}

create_plot_word(derecha, "marxista")
create_plot_word(derecha, "marxista")



x@docvars$n_words

tokens = editado2  %>% 
  filter(anio <= 1970) %>% 
  filter(!is.na(polo) & polo == "derecha") %>% 
  corpus() %>% 
  tokens(remove_punct = T, remove_symbols = T) %>%
  tokens_remove(stopwords) 



fcmat <- fcm(tokens, context = "window", tri = FALSE)
feat <- names(topfeatures(fcmat, 30))
feat <- setentas %>% 
  arrange(chi2) %>% 
  dplyr::slice(1:30) %>% 
  pull(feature)
  
fcm_select(fcmat, pattern = feat) %>%
  textplot_network()




dfm_noventas <- dfm %>% 
  dfm_subset(subset = setentas == 0) %>% 
  dfm_subset(subset = polo != "centro") 

correlacion_noventas <- textstat_keyness(dfm_noventas, target = dfm_noventas$polo == "izquierda")

textplot_keyness(correlacion_noventas)



dfm_2000 <- dfm %>% 
  dfm_subset(subset = setentas == 0) %>% 
  dfm_subset(subset = partido %in% c("partido socialista de chile", "partido union democrata independiente")) 

correlacion_2000 <- textstat_keyness(dfm_2000, target = dfm_2000$partido == "partido socialista de chile")

textplot_keyness(correlacion_2000, n = 30)


tstat_freq <- textstat_frequency(dfm, n = 5, groups = anio)
tstat_freq %>% 
ggplot(aes(x = reorder(feature, frequency), y = frequency)) +
  geom_point() +
  coord_flip() +
  labs(x = NULL, y = "Frequency") +
  facet_wrap(~group) +
  theme_minimal()

  