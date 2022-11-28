
# Cargar funciones necesarias para hacer las descargas
source("scripts/helpers.R")

library(XML)
library(feather)
library(furrr)
library(tictoc)
library(wnominate)
library(tidyverse)

clean_data <- read_feather("data/tabla_full_votaciones.feather")
nominate_anios <- read_feather("data/nominate_years.feather")


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

