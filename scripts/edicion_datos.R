##################
# EDICIÓN DE DATOS
##################

library(tidyverse)
library(feather)
library(lubridate)
library(writexl)
library(readxl)
library(data.table)
library(tictoc)
library(gender)
library(readxl)

# Cargar archivo de intervenciones
df <- read_feather("data/intervenciones.feather")

# Proviene de la tabla que me compartió NAcho con los políticos que aparecen relacionados con la palabra desigualdad
partidos <- read_feather("data/tabla_partidos.feather")

# Cargar archivo que contiene partidos de diputados. Este archivo fue creado por mí a partir de un xml que descargué de la página del congreso
partidos2 <- read_feather("data/tabla_militancias.feather")

# Cargar archivo que contiene partidos de senadores Este archivo fue creado por mí a partir de un xml que descargué de la página del congreso
senadores <- read_feather("data/tabla_militancias_senadores.feather")

# Cargar archivo con parlamentarios etiquetados por la hermana de Agloni
faltantes <- read_excel("data/partidos_faltantes_naj.xlsx")

# Cargar datos de biografias
bios <- read_feather("data/biografias.feather")

# Cargar tabla que contiene el sexo de cada uno de los parlamentarios
sexo_nombres <- read_feather("data/sexo_nombres.feather")

# Cargar tabla con los identificadores de todos los parlamentarios
id_politicos <- read_feather("data/identificadores_políticos.feather")


# Función para editar texto
edit_text <- function(text) {
  text_edit = tolower(text)
  text_edit = str_replace_all(text_edit, pattern = "é", "e")
  text_edit = str_replace_all(text_edit, pattern = "á", "a")
  text_edit = str_replace_all(text_edit, pattern = "í", "i")
  text_edit = str_replace_all(text_edit, pattern = "ó", "o")
  text_edit = str_replace_all(text_edit, pattern = "ú", "u")

  return(text_edit)
}




##################################################
# SELECCIONAR LAS INTERVENCIONES QUE SERÁN USADAS
#################################################


testing <- df %>%
  sample_n(1000)

mas_comunes <- testing %>%
  count(descripcion_tipo_participacion) %>%
  mutate(porcentaje = n /sum(n) * 100) %>%
  filter(porcentaje > 1)

filtro <- mas_comunes %>%
  pull(descripcion_tipo_participacion)

# testing %>%
#   filter(descripcion_tipo_participacion %in% filtro)  %>%
#   group_by(descripcion_tipo_participacion) %>%
#   slice(1:15) %>%
#   ungroup() %>%
#   arrange(descripcion_tipo_participacion) %>%
#   view()

# Se mantienen solo las intervenciones y adhesiones de los parlamentarios, porque son los que contienen mayor contenido ideológico. Al parecer, algunos archivos contienen información errónea respecto a la descripción de la participación, ya que las intervenciones por definición deberían tener solo una persona involucrada. Se agrega un filtro, para dejar solo las participaciones que tienen una persona.
filtrado <- df %>%
  filter(intervinientes == 1) %>%
  filter(descripcion_tipo_participacion %in% c("intervención", "adhesión"))  #%>% 
  #sample_n(10000)

################
# EDITAR TEXTOS
################


filtrado_dt <- data.table(filtrado %>%
                            mutate(index = row_number()))
setkey(filtrado_dt, index)


# Se hacen algunas depuraciones en los textos para tratar de limpiar un poco. No es posible eliminar toda la información superflua
# Es necesario depurar el identificador de algunos parlamentarios, ya que en algunos casos, existen 2 id diferentes para una misma persona, lo que genera que
# el match con las biografías fallen
tic()
filtrado_dt <- filtrado_dt[, texto_dep := str_remove_all(texto, "\\n|-")
                          ][, texto_dep := str_remove_all(texto_dep, "\\n")
                           ][, texto_dep := str_remove_all(texto_dep, "\nLa señora")
                             ][, texto_dep := str_replace_all(texto_dep, pattern = "  ", " ")
                             ][, texto_dep := str_replace_all(texto_dep, pattern = "  ", " ")
                               ][, texto_dep := tolower(texto_dep)
                                 ][, texto_dep := if_else(is.na(texto_dep), texto, texto_dep)
                                  ][, texto_dep := str_extract(string = texto_dep, "(señor|señora) president(e|a).+")
                                   ][, texto_dep := if_else(texto_dep == "señor presidente. \\n" | texto_dep == "señora presidenta. \\n" |  is.na(texto_dep) | nchar(texto_dep) < 50, tolower(texto), texto_dep)
                                     ][, list(id_doc, descripcion_tipo_participacion, texto, fecha, texto_dep, nombre, id, descripcion_debate, intervinientes, personas)
                                       ][, id := if_else(nombre == "Claudio Humberto Huepe García", 3988, as.numeric(id)  )
                                         ][, id := if_else(nombre == "Jorge Alfonso Burgos Varela", 1578, as.numeric(id))
                                           ][, id := if_else(nombre == "Carlos Hernán Bosselin Correa", 4083, as.numeric(id))]

toc()


########################
# Editar identificadores
########################

# En algunos casos, los textos se descargan con identificadores de personas erróneos. Se hace una conversión manual, para pegarle el verdadero identificador

# Lo primero que se hace es cambiar el identificador
filtrado_dt <- filtrado_dt %>%
  mutate(id = case_when(
    id_doc %in% c(2198487, 2173293, 2093726)  ~  4563, # confusión con Juan_Francisco_Undurraga_Gazitúa
    id_doc == 2173289                         ~  1438, # confusión con Fidel_Edgardo_Espinoza_Sandoval
    id_doc %in% c(2136210, 1789601)           ~  4530 , # confusión con Iván_Alberto_Flores_García
    id_doc %in% c(1883963, 1883965)           ~  -999,   # subsecreatario que no está presente en el listado de políticos
    id_doc %in% c(2173286, 2173294)           ~  4571,   # confusión con Miguel_Alejandro_Mellado_Suazo
    id_doc %in% c(1788629, 1850407,1788461)   ~  2100,   # confusión con Joaquín_Godoy_Ibáñez
    id_doc == 939503                          ~ 1434,   # Mario_Julio_Torres_Peralta
    T ~ id
    )) %>%
  mutate(cambio_id = if_else(id_doc %in% c(2198487, 2173293, 2093726, 2173289, 2136210, 1789601, 1883963, 1883965, 2173286, 2173294, 1788629, 1850407,1788461, 939503), 1, 0 ))


# Luego, le pegamos el nombre correcto, a partir del identificador corregido
filtrado_dt <- filtrado_dt %>%
  left_join(id_politicos %>%
              select(id_politico, nombre_completo) %>%
              mutate(id_politico = as.numeric(id_politico)),
            by = c("id" = "id_politico")) %>%
  mutate(nombre = if_else(nombre_completo != nombre & cambio_id == 1, nombre_completo, nombre) ) %>%
  select(-nombre_completo)


#########################
# Editar tabla biografías
##########################

# Arreglar algunas personas que quedaron sin partido. Se obtiene la información del partido a partir de un campo que contiene toda la info de cada político.
# Se deja el partido del último periodo parlamentario
bios_edit <- bios %>%
  mutate(partido2 = str_extract(datos_personales, pattern = "Partido.+\n"),
         partido2 = if_else(is.na(partido2) & str_detect(datos_personales, "Izquierda Radical"), "Izquierda Radical", partido2),
         partido2 = if_else(is.na(partido2) & str_detect(datos_personales, "Unión Demócrata Independiente"), "Partido Unión Demócrata Independiente", partido2),
         partido2 = if_else(is.na(partido2) & str_detect(datos_personales, "Independiente"), "Independiente", partido2),
         anio_inicio = str_extract(periodos, pattern = "[[:digit:]]{4}"), # extrar el año de inicio para cada periodo parlamentario
         anio_inicio = as.numeric(anio_inicio),
         partido = if_else(partido == "error" & !is.na(partido2), partido2, partido)
         ) %>%
  group_by(id_nombre) %>%
  arrange(desc(anio_inicio)) %>%
  slice(1) %>% # se deja el último periodo parlamentario
  ungroup() %>%
  mutate(id_numero = as.numeric(id_numero))

# Rescatar la información pegada en el campo de biografía. Se extrae profesión, fecha de nacimiento y de fallecimiento
bios_edit <- bios_edit %>%
  mutate(profesion = str_extract(datos_personales, pattern = "Profesión.+"),
         profesion = str_remove(profesion, "Profesión: "),
         nacimiento = str_extract(datos_personales, pattern = "Nacimiento.+\n"),
         nacimiento = str_extract(nacimiento, pattern = "de [[:digit:]]{4}"),
         nacimiento = str_remove(nacimiento, pattern = "de "),
         fallecimiento = str_extract(datos_personales, pattern = "Fallecimiento.+\n"),
         fallecimiento = str_extract(fallecimiento, pattern = "de [[:digit:]]{4}"),
         fallecimiento = str_remove(fallecimiento, pattern = "de ")
         )



# Encontrar el primer nombre para cada uno de los parlamentarios y hacer una depuración, para que la función de reconocimiento de sexo funcione bien
bios_edit <- bios_edit %>%
  mutate(primer_nombre = str_remove(id_nombre, "_.*"),
         primer_nombre = tolower(primer_nombre),
         primer_nombre = str_replace_all(primer_nombre, pattern = "á", "a"),
         primer_nombre = str_replace_all(primer_nombre, pattern = "é", "e"),
         primer_nombre = str_replace_all(primer_nombre, pattern = "í", "i"),
         primer_nombre = str_replace_all(primer_nombre, pattern = "ó", "o"),
         primer_nombre = str_replace_all(primer_nombre, pattern = "ú", "u"),
         primer_nombre = str_replace_all(primer_nombre, pattern = "ï", "i")

         )


# Obtener el sexo de los parlamentarios y pegar a la tabla de biografías
# nombres <- unique(bios_edit$primer_nombre)
# sexo_nombres <- gender(nombres, method = "genderize") # solo 1000 por día

bios_edit <- bios_edit %>%
  left_join(sexo_nombres %>%
              select(name, sexo = gender),
            by = c("primer_nombre" = "name"))

# Depurar la columna partido
bios_edit <- bios_edit %>%
  mutate(partido = tolower(partido)) %>%
  mutate(partido = str_remove_all(string = partido, pattern =  "\\n")) %>%
  mutate(partido = if_else(partido %in% c("partido radical de chile", "partido radical social demócrata",
                                          "partido radical socialdemócrata", "radical de chile"),
                           "partido radical", partido),
         partido = edit_text(partido),
         partido = if_else(partido == "renovacion nacional", "partido renovacion nacional", partido)         )

# Poner sexo a 2 casos para los que no se encontró etiqueta
bios_edit <- bios_edit %>%
  mutate(sexo = if_else(primer_nombre == "duberildo" | primer_nombre == "tucapel" , "male", sexo)) %>%
  mutate(sexo = case_when(
    sexo == "male" ~ "hombre",
    sexo == "female" ~ "mujer",
    T ~ NA_character_
  ) )


# Construir una tabla que identifica si la persona era miembro del Senado o de la Cámara en un periodo específico
camara_senado <-  bios %>% 
  mutate(periodo_numeric = str_extract(periodos, "[[:digit:]]{4}-[[:digit:]]{4}")) %>% 
  mutate(camara = tolower(periodos),
         camara= str_extract(camara, "diputado|senador|diputada|senadora") 
         ) %>% 
  mutate(nombre = str_replace_all(id_nombre, "_", " "),
         nombre = tolower(nombre)
         ) %>% 
  select(nombre, camara, periodo_numeric, id_nombre)
  

# Se guarda una tabla que permite saber a cuál de las dos cámaras pertenece cada político en un periodo determinado  
write_feather(camara_senado, "data/datos_camara.feather")

########################
# Unir textos con bios #
########################

# Agregar información relevante de cada político a la tabla de intervenciones
filtrado_dt_2 <- merge(filtrado_dt, bios_edit %>% select(id_numero, id_nombre, partido, sexo,
                                                         profesion, nacimiento, fallecimiento ),
                       all.x = TRUE, by.x = "id", by.y = "id_numero")


# Crear la edad que tenían las personas en el momento en el que se produjo la partición
filtrado_dt_2 <- filtrado_dt_2 %>%
  mutate(anio = as.numeric(str_sub(fecha, 1, 4)),
         nacimiento = as.numeric(nacimiento),
         edad_actual = anio - nacimiento
         )


#############################
# Crear periodos legislativos
#############################

filtrado_dt_2 <- filtrado_dt_2 %>%
  mutate(periodo_legislativo = case_when(
    anio <= 1969 ~ "1965-1969",
    anio <= 1973 ~ "1970-1973",
    anio <= 1994 ~ "1990-1994",
    anio <= 1998 ~ "1995-1998",
    anio <= 2002 ~ "1999-2002",
    anio <= 2006 ~ "2003-2006",
    anio <= 2010 ~ "2007-2010",
    anio <= 2014 ~ "2011-2014",
    anio <= 2018 ~ "2015-2018",
    anio <= 2022 ~ "2019-2022",
    anio <= 2026 ~ "2023-2026"

  ))

#######################################
# Extrar partidos de distintas fuentes
#######################################

partidos <- partidos %>%
  mutate(nombre_edit = edit_text(nombre))

partidos2_cortado <- partidos2 %>%
  mutate(nombre_edit = edit_text(nombre),
         anio = as.numeric(str_sub(fecha_inicio, 1, 4))) %>%
  group_by(nombre_edit) %>%
  arrange(desc(anio)) %>%
  slice(1) %>%
  ungroup()

senadores <- senadores %>%
  mutate(nombre_edit = edit_text(nombre))

faltantes <- faltantes %>%
  mutate(nombre_edit = edit_text(nombre))

partidos_politicos <- bind_rows(
  partidos %>% select(partido, nombre_edit),
  partidos2_cortado %>% select(partido = nombre_partido, nombre_edit),
  senadores %>% select(partido, nombre_edit),
  faltantes %>% select(partido, nombre_edit)
) %>%
  mutate(partido = edit_text(partido)) %>%
  filter(!is.na(partido)) %>%
  mutate(partido = case_when(
    partido %in% c("p.s.", "partido socialista")  ~ "partido socialista de chile",
    partido == "partido comunista" ~ "partido comunista de chile",
    partido == "revolucion democratica" ~ "partido revolucion democratica",
    partido == "u.d.i." ~ "partido union democrata independiente",
    partido %in% c("renovacion nacional", "r.n.")    ~ "partido renovacion nacional",
    partido == "p.p.d."   ~ "partido por la democracia",
    partido == "p.d.c."   ~ "partido democrata cristiano",
    partido == "evopoli" ~ "partido evolucion politica (evopoli)",
    partido == "union de centro-centro progresista" ~ "union de centro centro progresista",
    partido == "movimiento de izquierda cristiana" ~ "izquierda cristiana",
    partido == "independientes" ~ "independiente",

    T ~ partido
  )) %>%
  group_by(nombre_edit) %>%
  slice(1) %>%
  ungroup()

unique(partidos_politicos$partido) %>% sort()

# Se añada a la tabla principal el partido político de una conseguido a partir de varias fuentes.
# Se crean 3 variables para partido, dependiendo de la fuente
filtrado_dt_final <- filtrado_dt_2 %>%
  mutate(nombre_edit = edit_text(nombre)) %>%
  left_join(partidos_politicos,
            by = "nombre_edit") %>%
  rename(partido_descarga = partido.x,
         partido_varios = partido.y) %>%
  mutate(partido = partido_descarga,
         partido = if_else(partido %in% c("error", "independiente") & !is.na(partido_varios),
                           partido_varios, partido ))


unique(filtrado_dt_2$partido) %>% sort()
unique(filtrado_dt_final$partido) %>% sort()


#######################################
# Reparar partido de algunos políticos
#######################################

# Algunos políticos que cambiaron del partido nacional al partido de renovación nacional quedaron mal clasificados en los setenta.

filtrado_dt_final <- filtrado_dt_final %>%
  mutate(partido = case_when(
    id_nombre %in% c("Maximiano_Errázuriz_Eguiguren", "Gustavo_Alessandri_Valdés",
                     "Sergio_Eduardo_de_Praga_Diez_Urzúa", "Enrique_Larre_Asenjo",
                     "Francisco_Leandro_Bayo_Veloso", "Mario_Enrique_Ríos_Santander",
                     "Hugo_Álamos_Vásquez", "Osvaldo_Vega_Vera", "Sergio_Onofre_Jarpa_Reyes" ) &
      anio < 1990 ~ "partido nacional",
    T ~ partido

  ))


table(filtrado_dt_final$partido, filtrado_dt_final$anio)




#########################
# Ediciones específicas #
#########################
stopwords <- c(quanteda::stopwords("es"), "señor", "señores", "proyecto", "presidente", "presidenta", "dicho", "honorable", "don",
               "nyo", "garcia", "sanfuentes", "bulnes", "ibañez", "arnello", "ochagavia", "phillips", 
               "von", "muhlenbrock", "gustavo", "mühlenbrock", "nel", "solo", "sólo", "n°"
)

filtrado_dt_final <- filtrado_dt_final %>%
  mutate(setentas = if_else(anio <= 1973, 1, 0),
         partido2 = if_else(setentas == 1 & partido == "partido renovacion nacional", NA_character_, partido),
         polo = case_when(
           partido %in% c("partido socialista de chile", "partido comunista de chile", "partido convergencia social", "partido revolucion democratica",
                          "partido comunes", "izquierda radical", "partido humanista", "partido amplio de izquierda socialista", "union socialista popular",
                          "partido socialista popular", "izquierda cristiana", "revolucion democratica", "partido ecologista verde"
                          ) ~ "izquierda",
           partido %in% c("partido nacional", "partido union democrata independiente", "partido renovacion nacional", "partido republicano de chile",
                          "partido conservador unido", "partido conservador", "partido evolucion politica (evopoli)", "amplitud", "union democrata independiente",
                          "partido nacional o monttvarista"
                          
                          ) ~ "derecha",
           partido %in% c("partido radical", "partido democrata cristiano", "partido por la democracia", "partido radical socialdemocrata") ~ "centro"
         )
  ) %>%
  mutate(texto_dep = str_remove_all(texto_dep, pattern = "-")) %>%
  mutate(periodo_legislativo_label = as.factor(periodo_legislativo)) %>% 
  mutate(periodo_legislativo_label = fct_relevel(periodo_legislativo_label, c("1965-1969", 
                                                                              "1970-1973", "1990-1994",
                                                                              "1995-1998", "1999-2002", "2003-2006", "2007-2010", "2011-2014",
                                                                              "2015-2018", "2019-2022"))) 
write_feather(filtrado_dt_final, "data/edicion_inicial.feather")

# Guardar versión light con algunas filas
set.seed(123)
filtrado_dt_final <- filtrado_dt_final %>% 
  sample_n(100000)

# Guardar datos para la investigación
write_feather(filtrado_dt_final, "data/edicion_inicial_light.feather")



x2 <- x %>% 
  left_join()


