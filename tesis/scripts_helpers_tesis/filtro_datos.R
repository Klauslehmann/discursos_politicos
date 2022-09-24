
library(tidyverse)
library(feather)

# Cargar archivo de intervenciones
df <- read_feather("data/intervenciones.feather")
filtrado <- read_feather("data/edicion_inicial.feather") 

# Construir tabla de datos con el resumen de las filas antes y después de hacer el filtro 
cantidad_filas <-  tribble(
                          ~ "filtro",              ~ "cantidad de filas",
                            "datos brutos",           nrow(df),
                            "datos filtrados",        nrow(filtrado)
)

# Construir tabla que muestra la cantidad de datos por año después de haber filtrado
agregar_na <- data.frame(anio = 1974, n = NA)

n_year <-  filtrado %>% 
  group_by(anio) %>% 
  summarise(n = n()) %>% 
  bind_rows(agregar_na)

# Guardar tablas para ser usadas en el texto
write_feather(n_year, "data/cuadros_tesis/intervenciones_anio.feather")
write_feather(cantidad_filas, "data/cuadros_tesis/cantidad_filas.feather")
