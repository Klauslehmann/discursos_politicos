library(tidyverse)

# Cargar datos
score <- arrow::read_feather("data/score_full.feather")


# Mirar indicador a lo largo del tiempo
score %>% 
  summarise(media = mean(score),
            min = min(score),
            max = max(score),
            median = median(score)
            )

score %>% 
  filter(n_words >= 1) %>% 
  group_by(anio) %>% 
  summarise(media = mean(score) ) %>% 
  bind_rows(data.frame(anio = 1974, media = NA)) %>% 
  mutate(ma = zoo::rollmean(media, k = 3, fill = NA)) %>% 
  ggplot(aes(x = anio, y = ma, group = 1)) +
  geom_line() +
  geom_point() +
  geom_vline(xintercept = 1970, linetype = "dashed") +
  geom_vline(xintercept = 1973, linetype = "dashed") +
  geom_vline(xintercept = 1990, linetype = "dashed") +
  geom_vline(xintercept = 1994, linetype = "dashed") +
  geom_vline(xintercept = 2000, linetype = "dashed") +
  geom_vline(xintercept = 2006, linetype = "dashed") +
  geom_vline(xintercept = 2010, linetype = "dashed") +
  geom_vline(xintercept = 2014, linetype = "dashed") +
  geom_vline(xintercept = 2018, linetype = "dashed") +
  geom_vline(xintercept = 2022, linetype = "dashed")







x <- score %>% 
  filter(anio >= 1990) %>% 
  mutate(fecha = lubridate::as_date(fecha) ) %>% 
  group_by(fecha) %>% 
  summarise(media = mean(score) ) %>% 
  mutate(ma = zoo::rollmean(media, k = 30, fill = NA)) %>% 
  #bind_rows(data.frame(fecha = lubridate::as_date("1973-09-12"), media = NA) ) %>% 
  ggplot(aes(x = fecha, y = media, group = 1)) +
  geom_line() +
  geom_smooth()
  scale_x_date(date_breaks = "5 month") +
  geom_vline(xintercept = as.Date( "1970-11-03"), linetype = "dashed", color = "red") +
  geom_vline(xintercept = as.Date( "1990-03-11"), linetype = "dashed", color = "red") +
  geom_vline(xintercept = as.Date( "1994-03-11"), linetype = "dashed", color = "red") + 
  geom_vline(xintercept = as.Date( "2000-03-11"), linetype = "dashed", color = "red") + 
  geom_vline(xintercept = as.Date( "2006-03-11"), linetype = "dashed", color = "red") + 
  geom_vline(xintercept = as.Date( "2010-03-11"), linetype = "dashed", color = "red") + 
  geom_vline(xintercept = as.Date( "2014-03-11"), linetype = "dashed", color = "red") + 
  geom_vline(xintercept = as.Date( "2018-03-11"), linetype = "dashed", color = "red") + 
  geom_vline(xintercept = as.Date( "2022-03-11"), linetype = "dashed", color = "red") + 
  theme(axis.text.x = element_text(angle = 90, size = 5))

