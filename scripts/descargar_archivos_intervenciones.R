
library(rjson)
library(purrr)

# Cargar funciones necesarias para hacer las descargas
source("scripts/helpers.R")

# Extraer listado de ids. El archivo listado_politicos.json se descarga manualmente de la página de la biblioteca del congreso. No recuerdo cómo llegué a este solución. El archivo json está en la siguiente url: https://www.bcn.cl/laborparlamentaria/wsgi/consulta/FacetasBuscadorAvanzado.py
listado <- rjson::fromJSON(file = "data/listado_politicos_actualizado.json")
politicos <- listado$personas

id_politicos <- map_chr(politicos, ~.x$id )
nombres_id <-  map(politicos, ~.x$id_biografias_bcn)
nombres_id[map_lgl(nombres_id, is.null) ] <- "sin dato"
nombre_completo <- map_chr(politicos, possibly(~.x$descripcion, otherwise = "error")  )

identificadores <- data.frame(id_politico = id_politicos, nombre_id = unlist(nombres_id), nombre_completo = nombre_completo)



# Descargar los archivos json con la información de los políticos.
for (id in 1:length(id_politicos)) { #length(id_politicos)
  url <- paste0("https://www.bcn.cl/laborparlamentaria/wsgi/consulta/FacetasBuscadorAvanzado.py?sort=desc&personas=",
                id_politicos[id], "&start=0&rows=2000")
  download.file(url = url, destfile = paste0("data/historia_politicos/file", id_politicos[id], ".json") )
  print(id)
}




for (id in identificar_indices_corruptos(e$file)) {
  url <- paste0("https://www.bcn.cl/laborparlamentaria/wsgi/consulta/FacetasBuscadorAvanzado.py?sort=desc&personas=",
                id_politicos[id], "&start=0&rows=2000")
  download.file(url = url, destfile = paste0("data/historia_politicos/file", id_politicos[id], ".json") )
  print(id)
}


# Ejemplo de descarga de una url
url <- paste0(
  "https://www.bcn.cl/laborparlamentaria/wsgi/consulta/FacetasBuscadorAvanzado.py?sort=desc&personas=",
  1, "&start=0&rows=2000"
  )

download.file(url = url, destfile = paste0("data/historia_politicos/file", id_politicos[568], ".json") )
