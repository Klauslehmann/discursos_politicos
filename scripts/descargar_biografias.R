
library(RSelenium)
library(wdman)
library(rvest)
library(tidyverse)
library(furrr)
library(tictoc)
library(feather)

# Cargar funciones
source("scripts/helpers.R")


# Extraer listado de ids. El archivo listado_politicos.json se descarga manualmente de la página de la biblioteca del congreso. No recuerdo cómo llegué a este solución. El archivo json está en la siguiente url: https://www.bcn.cl/laborparlamentaria/wsgi/consulta/FacetasBuscadorAvanzado.py
listado <- rjson::fromJSON(file = "data/listado_politicos_actualizado.json")
politicos <- listado$personas

id_politicos <- map_chr(politicos, ~.x$id )
nombres_id <-  map(politicos, ~.x$id_biografias_bcn)
nombres_id[map_lgl(nombres_id, is.null) ] <- "sin dato"
nombre_completo <- map_chr(politicos, possibly(~.x$descripcion, otherwise = "error")  )

identificadores <- data.frame(id_politico = id_politicos, nombre_id = unlist(nombres_id), nombre_completo = nombre_completo)



#Se utiliza el paquete RSelenium.
#Esto es para ver la versión del driver de Chrome. No entiendo bien lo de las versiones. Solo una funciona.
version <- binman::list_versions("chromedriver")

#Conectarse al servidor de Selenium. Se usa la sexta versión que aparece en la lista de más arriba. No entiendo por qué funciona esta y no las demás. Al parecer tiene relación con las actualizaciones del browser. Probablemente en el futuro sea necesario hacer ajustes a la versión.
rD1 <- rsDriver(browser = "chrome", port = 45673L, geckover = NULL,
                chromever = version[[1]][17], iedrver = NULL, phantomver = NULL)

remDr <- rD1[["client"]]

# Setear 1.5 segundos de espera como máximo
remDr$setTimeout(type = "implicit", milliseconds = 1500)


# Estas dos listas corresponden a los identificadores de los parlamentarios. Se construyen a partir de un archivo cargado más arriba.
# Se deben hacer algunas correcciones manuales, debido a problemas en los identificadores de algunos políticos
ids_nombres <-  identificadores %>%
  mutate(nombre_id = if_else(nombre_id == "Nicanor_Araya_De_la_Cruz", "Nicanor_De_la_Cruz_Araya", nombre_id)) %>%
  mutate(nombre_id = if_else(nombre_id == "Alonso_Zumaeta_Faúndez", "Alonso_Zumaeta_Faune", nombre_id)) %>%
  mutate(nombre_id = if_else(nombre_id == "Joaquín_Palma_Irarrázaval", "Joaquín_Salvador_José_María_Antonio_Palma_Irarrázaval", nombre_id)) %>%
  mutate(nombre_id = if_else(nombre_id == "Juan_Bautista_Segundo_Argandoña_Cortés", "Juan_Bautista_Segundo_Argandoña_Cortéz", nombre_id)) %>%
  filter(nombre_id != "sin dato" & nombre_id != "") %>%
  pull(nombre_id)


ids_numericos <-  identificadores %>%
  mutate(id_politico = if_else(nombre_id == "Carlos_Hernán_Bosselin_Correa", "4083", id_politico )) %>%
  mutate(id_politico = if_else(nombre_id == "Claudio_Humberto_Huepe_García", "3988", id_politico )) %>%
  filter(nombre_id != "sin dato" & nombre_id != "") %>%
  pull(id_politico)

# Se van descargando los datos biográficos a partir del listado de indetificadores
n <- length(ids_nombres)
tic()
datos <- map2(ids_nombres[1:n], ids_numericos[1:n], possibly(~extraer_biografia(.x, .y), otherwise = "error") )
length(datos)
toc()


# Identificar los casos que fallaron en la descarga
detectar_errores()


# Seleccionar todo lo que no tuvo problemas durante la descarga
final <- datos[datos != "error"] %>%
  bind_rows()

# Escribir una tabla feather
write_feather(final, "data/clean_data/biografias.feather")

# Escribir una tabla con los identificadores para usarla más tarde
write_feather(identificadores, "data/clean_data/identificadores_políticos.feather")


#Cerrar servidor y el puerto
remDr$close()
rD1[["server"]]$stop()
system("taskkill /im java.exe /f", intern=FALSE, ignore.stdout=FALSE)
