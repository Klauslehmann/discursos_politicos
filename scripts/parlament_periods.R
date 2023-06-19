library(tidyverse)
library(feather)

# Cargar archivos
bios <- read_feather("data/datos_camara.feather")
bios <- bios %>% 
  mutate(id = row_number()) %>% 
  mutate(id = paste0(id_nombre, id))

# Separar todas las filas del dataframe para trabajar los datos con un formato de lista
bios_list <- bios %>% 
  filter(!is.na(periodo_numeric)) %>% 
  split(~id)

# Función que genera un dato por año para cada diputado
expand_year <- function(period) {
  years <- as.numeric(str_split(period$periodo_numeric, "-")[[1]])
  range <-   years[1]:years[2]
  return(list("name" = period$nombre, "role" = period$camara, "range" = range ))    
  
}

# Unir todo en un dataframe. Esta tabla contiene el rol de cada diputado por año
data <- map(bios_list, expand_year) %>% 
  map(as.data.frame) %>% 
  bind_rows() %>% 
  rename(anio = range)

# Descartar las filas totalmente repetidas. No tiene sentido que se repita nombre, año y rol. Hacer este filtro no genera perdida de informacion
data_sin_dup <- data %>% 
  group_by(name, anio, role) %>% 
  slice(1) %>% 
  ungroup()
  

# Identificar si existen duplicados. Los datos duplicados corresponden a personas que transitaron entre diputado y senador. Se decide mantener ambos registros y duplicar los textos que se encuentran en esta condición. 
data_sin_dup %>% 
  group_by(name, anio) %>% 
  mutate(contar = n()) %>% 
  ungroup() %>% 
  filter(contar > 1 & anio >= 1965) %>% 
  view()

# Guardar datos
write_feather(data_sin_dup, "data/periodos_parlamentarios.feather")




