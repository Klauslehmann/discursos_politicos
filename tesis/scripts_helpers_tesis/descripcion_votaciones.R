


library(tidyverse)
library(feather)
library(lubridate)

# Cargar archivo de votaciones descargado mediante la API de la cámara de diputados
df <- read_feather("data/tabla_full_votaciones.feather")

# Contar votaciones por año
df <- df %>% 
  mutate(fecha2 = as_date(fecha),
         year = year(fecha2)
         )

votaciones_anio <- df %>% 
  group_by(id_votacion) %>% 
  slice(1) %>% 
  group_by(year) %>% 
  summarise(n = n()) %>% 
  ungroup()


write_feather(votaciones_anio, "tesis/cuadros_tesis/votaciones_anio.feather")
