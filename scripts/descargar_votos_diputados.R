library(XML)
library(tidyverse)
library(feather)
library(furrr)
library(tictoc)
library(wnominate)
library(lubridate)

# El script contiene algunas funciones que permiten trabajar con la API de la cámara de diputados.
# Trabajo en desarrollo.

# Cargar funciones necesarias para hacer las descargas
source("scripts/helpers.R")

#########################################
# Ejemplo de uso para un año específico
########################################


# COnseguir los id de cada votación. Se guardaen un dataframe todos los id de las votaciones
anios <- 2002
id_votaciones <-  map_df(anios, descargar_id_votaciones)
id_votaciones_unicos <- id_votaciones %>%
  distinct()

# Descagar el detalle de votos  para cada votación
plan(multisession, workers = 8)
votaciones <- future_map(id_votaciones_unicos$Id,
                    possibly(~descargar_detalle_votacion(.x), otherwise = "error"))

# Sacar algunos casos fallidos
which(votaciones == "error")
clean_data <-  votaciones[votaciones != "error"] %>%
  bind_rows()

################################
# Buscar militancia de políticos
################################

# Conseguir la militancia de los partidos. Esta tabla contiene todas las militancias de los diputados
militancia_diputados <- get_deputy_parties(party = "all")

###################################
# Generar formato para w-nominate #
###################################

# Generar el formato necesario para el paquete creado por Keith Poole
roll_call_data <-  create_vote_data(clean_data, militancia_diputados)

##############
# W-NOMINATE #
##############



# voto 0 = en contra
# voto 1 = afirmativo
# voto 2 = abstención
# voto 3 = dispensado

# Crear objeto para calcular NOMINATE
rc <- rollcall(roll_call_data$votes_matrix, yea=c(1), nay=c(0),
                missing=c(2, 3),notInLegis=4, legis.names=roll_call_data$legislator_names, legis.data=roll_call_data$party, desc="UN Votes" )

# Crear la polaridad
polarity <- create_polarity(roll_call_data)

# Calcular primeras 2 dimensiones de NOMINATE
result <- wnominate(rc, polarity=polarity)

# Crear un gráfico con el promedio por partido en la primera dimensión
plot_mean_party(result)



##################################
# W-NOMINATE PARA TODO EL PERIODO
##################################

# El código que viene más abajo calcula NOMINATE para todos los años disponibles
# Calcular las dimensiones de nomininate para todo el periodo

tic()
nominate <- map(2002:2022, function(x) {
  nom_input <- create_nominate_input(x)
  print(x)
  return(nom_input)
} )
toc()


# Construir una tabla para todos los años con los valores de las primeras 2 dimensiones
nominate_anios <- map(nominate, ~.x$legislators %>%
      mutate(year = .x$year)
      ) %>%
  bind_rows()


#############
# Graficar #
#############

# Gráfico de puntos por año
nominate_anios %>%
  select(coord1D, party, year) %>%
  arrange(coord1D) %>%
  filter(!is.na(coord1D)) %>%
  filter(party %in% c("PCS", "UDI", "RN", "PC", "PS", "PPD", "DC", "RD", "EVOP", "PREP")) %>%
  group_by(party, year) %>%
  mutate(coord1D = mean(coord1D)) %>%
  slice(1) %>%
  ungroup() %>%
  ggplot(aes(x =  reorder(party, coord1D ),  y = coord1D, color = party, label = party)) +
  geom_point(size = 2) +
  geom_text(hjust=1.5, vjust=0) +
  facet_wrap(~year) +
  coord_flip() +
  labs(title = "W-NOMINATE para votaciones (media partidos)",
       subtitle = "primera dimensión")

ggsave("graficos/wnominate_patidos_varios_anios.pdf", width = 17, height = 12)

# Gráfico de líneas
nominate_anios %>%
  select(coord1D, party, year) %>%
  arrange(coord1D) %>%
  filter(!is.na(coord1D)) %>%
  filter(party %in% c("PCS", "UDI", "RN", "PC", "PS", "PPD", "DC", "RD", "EVOP", "PREP")) %>%
  group_by(party, year) %>%
  mutate(coord1D = mean(coord1D)) %>%
  slice(1) %>%
  ungroup() %>%
  ggplot(aes(x =  as.factor(year),  y = coord1D, color = party, group = party)) +
  geom_line() +
  geom_point() +
  labs(title = "W-NOMINATE para votaciones (media partidos)",
       subtitle = "primera dimensión")

ggsave("graficos/wnominate_patidos_varios_anios_lineas.pdf", width = 17, height = 12)




# Guardar identificadores de votaciones
write_feather(nominate_anios, "data/clean_data/nominate_years.feather")

