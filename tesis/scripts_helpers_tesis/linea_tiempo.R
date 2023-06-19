library(tidyverse)
library(feather)

# Cargar datos
#score <- arrow::read_feather("data/score_full.feather")
score <- arrow::read_feather("data/score_filtered.feather")

years <- c(1974:1989)
months <- c(1:12)
date <- c()
for (year in years) {
  for (month in months)  {
    new_date <- paste(year, month, sep = "-")
    date <- c(date, new_date)
  }
}

date <- date[11: length(date)]



tabla_timeline <-  score %>% 
  filter(n_words >= 1) %>% 
  mutate(fecha = lubridate::as_date(fecha),
         mes = lubridate::format_ISO8601(fecha, precision = "ym")  ) %>% 
  group_by(mes) %>% 
  summarise(media = mean(score) ) %>% 
  mutate(ma = zoo::rollmean(media, k = 10, fill = NA)) %>% 
  bind_rows(data.frame(mes = date, ma = rep(NA, length(date)) )) %>% 
  mutate(mes = lubridate::ym(mes)) 

write_feather(tabla_timeline, "tesis/cuadros_tesis/tabla_timeline.feather")


tabla_timeline_senadores <-  score %>% 
  filter(n_words >= 1 & role == "senador") %>% 
  mutate(fecha = lubridate::as_date(fecha),
         mes = lubridate::format_ISO8601(fecha, precision = "ym")  ) %>% 
  group_by(mes) %>% 
  summarise(media = mean(score) ) %>% 
  mutate(ma = zoo::rollmean(media, k = 10, fill = NA)) %>% 
  bind_rows(data.frame(mes = date, ma = rep(NA, length(date)) )) %>% 
  mutate(mes = lubridate::ym(mes)) 

write_feather(tabla_timeline_senadores, "tesis/cuadros_tesis/tabla_timeline_senadores.feather")


tabla_timeline_diputados <-  score %>% 
  filter(n_words >= 1 & role == "diputado") %>% 
  mutate(fecha = lubridate::as_date(fecha),
         mes = lubridate::format_ISO8601(fecha, precision = "ym")  ) %>% 
  group_by(mes) %>% 
  summarise(media = mean(score) ) %>% 
  mutate(ma = zoo::rollmean(media, k = 10, fill = NA)) %>% 
  bind_rows(data.frame(mes = date, ma = rep(NA, length(date)) )) %>% 
  mutate(mes = lubridate::ym(mes)) 

write_feather(tabla_timeline_diputados, "tesis/cuadros_tesis/tabla_timeline_diputados.feather")



tabla_timeline_diputados_derecha <-  score %>% 
  filter(n_words >= 1 & role == "diputado" & polo == "derecha") %>% 
  mutate(fecha = lubridate::as_date(fecha),
         mes = lubridate::format_ISO8601(fecha, precision = "ym")  ) %>% 
  group_by(mes) %>% 
  summarise(media = mean(score) ) %>% 
  mutate(ma = zoo::rollmean(media, k = 10, fill = NA)) %>% 
  ungroup() %>% 
  bind_rows(data.frame(mes = date, ma = rep(NA, length(date)) )) %>% 
  mutate(mes = lubridate::ym(mes),
         polo = "derecha"
         ) 

write_feather(tabla_timeline_diputados_derecha, "tesis/cuadros_tesis/tabla_timeline_diputados_derecha.feather")


tabla_timeline_diputados_izquierda <-  score %>% 
  filter(n_words >= 1 & role == "diputado" & polo == "izquierda") %>% 
  mutate(fecha = lubridate::as_date(fecha),
         mes = lubridate::format_ISO8601(fecha, precision = "ym")  ) %>% 
  group_by(mes) %>% 
  summarise(media = mean(score) ) %>% 
  mutate(ma = zoo::rollmean(media, k = 10, fill = NA)) %>% 
  ungroup() %>% 
  bind_rows(data.frame(mes = date, ma = rep(NA, length(date)) )) %>% 
  mutate(mes = lubridate::ym(mes),
         polo = "izquierda") 

write_feather(tabla_timeline_diputados_izquierda, "tesis/cuadros_tesis/tabla_timeline_diputados_izquierda.feather")


timeline_diputados_polos <- tabla_timeline_diputados_derecha %>% 
  bind_rows(tabla_timeline_diputados_izquierda)

write_feather(timeline_diputados_polos, "tesis/cuadros_tesis/timeline_diputados_polos.feather")





affect <- score %>% 
  filter(n_words >= 1) %>% 
  mutate(fecha = lubridate::as_date(fecha),
         mes = lubridate::format_ISO8601(fecha, precision = "ym")  ) %>% 
  group_by(mes) %>% 
  summarise(media = mean(cos_affect) ) %>% 
  mutate(ma = zoo::rollmean(media, k = 10, fill = NA)) %>% 
  bind_rows(data.frame(mes = date, ma = rep(NA, length(date)) )) %>% 
  mutate(mes = lubridate::ym(mes)) %>% 
  mutate(var = "afectivo")


cog <-  score %>% 
  filter(n_words >= 1) %>% 
  mutate(fecha = lubridate::as_date(fecha),
         mes = lubridate::format_ISO8601(fecha, precision = "ym")  ) %>% 
  group_by(mes) %>% 
  summarise(media = mean(cos_cognitive) ) %>% 
  mutate(ma = zoo::rollmean(media, k = 10, fill = NA)) %>% 
  bind_rows(data.frame(mes = date, ma = rep(NA, length(date)) )) %>% 
  mutate(mes = lubridate::ym(mes)) %>% 
  mutate(var = "cognitivo")

final <- affect %>% 
  bind_rows(cog)

final %>% 
  ggplot(aes(x = mes, y = ma, group = var, color = var)) +
  geom_line()






####################################################
# Generar línea de tiempo para tendencias políticas
####################################################

data <-  score %>% 
  filter(!is.na(polo)) %>% 
  mutate(fecha = lubridate::as_date(fecha),
         mes = lubridate::format_ISO8601(fecha, precision = "ym")  ) %>% 
  group_by(anio, polo) %>% 
  summarise(conteo = n(),
            media = mean(score)
            ) %>% 
  ungroup() 



smoothed <- hpfilter(as.ts(data$media), drift = T)

test <- data.frame(media = as.vector(smoothed$trend)) %>% 
  mutate(anio = data$mes)


data  %>%
  ggplot(aes(anio, media, group = polo, color = polo)) +
  geom_line() +
  geom_point()


library(mFilter)
data(unemp)

opar <- par(no.readonly=TRUE)
unemp.hp1 <- hpfilter(unemp, drift=TRUE)
unemp.hp2 <- hpfilter(unemp, freq=800, drift=TRUE)
unemp.hp3 <- hpfilter(unemp, freq=12,type="frequency",drift=TRUE)
unemp.hp4 <- hpfilter(unemp, freq=52,type="frequency",drift=TRUE)

par(mfrow=c(2,1),mar=c(3,3,2,1),cex=.8)
plot(unemp.hp1$x,  ylim=c(2,13),
     main="Hodrick-Prescott filter of unemployment: Trend, drift=TRUE",
     col=1, ylab="")
lines(unemp.hp1$trend,col=2)
lines(unemp.hp2$trend,col=3)
lines(unemp.hp3$trend,col=4)
lines(unemp.hp4$trend,col=5)
legend("topleft",legend=c("series", "lambda=1600", "lambda=800",
                          "freq=12", "freq=52"), col=1:5, lty=rep(1,5), ncol=1)


tramos <-  read_csv("/home/klaus/Downloads/tramos.csv")
write_csv(tramos, "/home/klaus/Downloads/tramos.csv")