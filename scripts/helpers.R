
#id_nombre: identificador de nombre para un político
# id_numero: identificador numérico para un político
# Busca en la página del congreso nacional la biografía de cada político. Descarga algunos campos de interés
# Devuelve un data frame com algunos datos del parlamentario

extraer_biografia <- function(id_nombre, id_numero) {
  # id_nombre <- "Tulio_Renán_Fuentealba_Moena"
  # id_numero <- 1

  url <- paste0("https://www.bcn.cl/historiapolitica/resenas_parlamentarias/wiki/", id_nombre)

  remDr$navigate(url)

  # Periodos parlamentarios
  periodos <- remDr$findElements(using = 'xpath', '//*[@id="trayectoria_new"]/tbody/tr/td/div[1]')
  periodos <- map(periodos, ~.x$getElementText() ) %>%
    purrr::flatten()

  periodos <- periodos[!periodos %in% c("Presidenta de la Cámara de Diputados", "Presidenta del Senado",
                                        "Presidente de la Cámara de Diputados", "Presidente del Senado", "",
                                        "Intendente de Región de Antofagasta")  ]

  # periodos 2.0
  periodos2 <- remDr$findElements(using = 'xpath', '//*[@id="trayectoria_new"]/tbody/tr/td/div[1]')
  periodos2 <- map(periodos2, ~.x$getElementText() ) %>%
    purrr::flatten()

  periodos2 <- periodos2[!periodos2 %in% c("Presidenta de la Cámara de Diputados", "Presidenta del Senado",
                                           "Presidente de la Cámara de Diputados", "Presidente del Senado", "",
                                           "Intendente de Región de Antofagasta"
  )  ]


  # Partido
  partido <- remDr$findElements(using = 'xpath', '//*[@id="trayectoria_new"]/tbody/tr/td/div[3]/span')
  partido <- map(partido, ~.x$getElementText() ) %>%
    purrr::flatten()

  partido <- partido[!partido %in% c("", "8ª Circunscripción", '\"Tercera Agrupación Provincial, Valparaíso y Aconcagua\"',
                                     "Sexta Agrupación Departamental \"Valparaíso y Quillota\""
  )]

  partido <- partido[!str_starts(partido, "[[:digit:]]") &  !str_starts(partido, "Distrito") &
                       !str_starts(partido, "Primera|Segunda|Tercera|Cuarta|Quinta|Sexta|Séptima|Octava|Novena|Décima|Decimoséptima|Agrupación Provincial") ]

  # Dejar el último partido cuando el largo del vector no coincide con los periodos parlamentarios. Esto es solo para no perder información
  # Si no se extrae ningún partido es porque hay un error
  if (length(partido) == 0)  {
    partido <- "error"
  } else if  (length(partido) < length(periodos)) {
    partido <- partido[1] # Se deja el último partido en el que militó
  }


  #  Datos personales. Se obtiene la tabla completa, para luego extraer la información relevante
  datos_personales <- remDr$findElements(using = 'xpath', '//*[@id="trayectoria_new"]')
  datos_personales <- datos_personales[[2]]$getElementText()[[1]]


  tabla_datos <- data.frame(periodos = unlist(periodos),
                            periodos2 = unlist(periodos2),
                            #cuenta_twitter = twitter,
                            partido = unlist(partido),
                            datos_personales = datos_personales,
                            id_nombre = id_nombre,
                            id_numero = id_numero)


  return(tabla_datos)
}



# Detectar errores.
# Procedimiento que usa variables de entorno para identificar la cantidad de biografías que no se descargaron
# Devuelve una lista con los identificadores de nombre y numéricos
detectar_errores <- function() {
  fail_index <-  which(datos == "error")
  return(list(ids_nombres[fail_index], ids_numericos[fail_index]))

}

# Recibe un año y devuelve los datos de todas las votaciones de ese año
descargar_id_votaciones <-  function(anio) {
  url <-   paste0("http://opendata.camara.cl/camaradiputados/WServices/WSLegislativo.asmx/retornarVotacionesXAnno?prmAnno=", anio)

  dataframe <- xmlToDataFrame(url)

  dataframe


}
# Recibe un identificador de votación y devuelve la votación de cada diputado
descargar_detalle_votacion <- function(id_votacion) {


  root <- "http://opendata.camara.cl/camaradiputados/WServices/WSLegislativo.asmx/retornarVotacionDetalle?prmVotacionId="

  url <- paste0(root, id_votacion)


  xml <- xmlTreeParse(url, useInternal = T)
  xml_list <- xmlToList(xml)


  votos_diputados <-  map_df(xml_list$Votos, ~bind_cols(.x$Diputado %>% as.data.frame(),
                                                        .x$OpcionVoto %>% as.data.frame(),
                                                        id_votacion = id_votacion) ) %>%
    mutate(fecha = xml_list$Fecha )

  return(votos_diputados)

}

# Función para obtener la militancia de todos los parlamentarios
# Recibe un string que indica si se deben devolver todas las militancias o solo la más actual

get_deputy_parties<- function(party = "last") {

  # Consultamos los datos de todos los diputados
  url <- "http://opendata.camara.cl/camaradiputados/WServices/WSDiputado.asmx/retornarDiputados?"
  xml <- xmlTreeParse(url, useInternal = T)
  xml_list <- xmlToList(xml)

  # Crear una lista inicial con las militancias de todos los parlamentarios
  militancias <- map(xml_list, ~.x$Militancias)

  # Buscar datos que identifican a los diputados
  nombres <- map_chr(xml_list, ~.x$Nombre)
  id <- map_chr(xml_list, ~.x$Id)
  apellido <- map_chr(xml_list, ~.x$ApellidoPaterno)

  # Seleccionar fechas partido y alias de partido
  fechas_inicio <- map(militancias, function(z) { map_chr(z, 1)} )
  partidos <- map(militancias, function(z) { map_chr(z, ~.x$Partido$Alias)} )
  partidos_completo <- map(militancias, function(z) { map_chr(z, ~.x$Partido$Nombre)} )


    # Get last party for each one
  if (party == "last") {
    datos <-  pmap_df(list(id, partidos, partidos_completo, fechas_inicio, nombres, apellido) ,
                      function(id, p, p2, f, n, a) { bind_cols(id = id, partido = p, partido_completo = p2,
                                                         fecha = f, nombre = n, apellido = a ) } ) %>%
      group_by(id) %>%
      arrange(desc(fecha)) %>%
      slice(1) %>%
      ungroup()

  } else if (party == "all") {
    datos <-  pmap_df(list(id, partidos,partidos_completo, fechas_inicio, nombres, apellido) ,
                      function(id, p, p2, f, n, a) { bind_cols(id = id, partido = p, partido_completo = p2,
                                                         fecha = f, nombre = n, apellido = a ) } )

  }


  return(datos)
}



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


# Función para generar el formato de votos que necesita el paquete
# Recibe un dataframe con votos en formato long
# Devuelve una lista con 3 objetos:
# - Matriz de votos
# - Listado con el nombre de los diputados
# - Listado de partidos al que pertence cada diputado

#militancia_diputados <- get_deputy_parties(party = "all")


create_vote_data <- function(long_data, militancia_diputados) {

  #long_data <- clean_data

  # Dejar en formato requerido para nominate
  wide_clean_data <- long_data %>%
    mutate(nombre_completo = paste(Nombre, ApellidoPaterno, ApellidoMaterno, sep = "_"),
           nombre_completo = tolower(nombre_completo),
           nombre_completo = str_replace_all(nombre_completo, " ", "_"),
           nombre_completo = edit_text(nombre_completo),
           voto = .attrs) %>%
    pivot_wider(id_cols = c("nombre_completo", "Id"), names_from = "id_votacion", values_from = "voto",
                names_prefix = "v")  %>%
    mutate(fecha = long_data$fecha[1]) %>%
    # Se deja la militancia más cercana a la fecha de votación
    left_join(militancia_diputados,
              by = c("Id" = "id" )) %>%
    mutate(dif = abs(as_datetime(fecha.x) - as_datetime(fecha.y) ) ) %>%
    group_by(Id) %>%
    arrange(dif) %>%
    slice(1) %>%
    ungroup()

  # Los nombres de los diputados se usan más adelante
  names <- wide_clean_data %>%
    pull(nombre_completo)

  # Los datos de partido también se usan más adelante
  legData <- wide_clean_data %>%
    select(party = partido)

  # Matriz que contiene todas las votaciones de los diputados. Es el formato requerido por el paquete
  mat_votos <- wide_clean_data %>%
    select(-nombre_completo, -nombre, -apellido, -partido, -partido_completo, -starts_with("fecha"), -Id, -dif ) %>%
    mutate_all(as.integer) %>%
    mutate_all(.funs =  ~if_else(is.na(.), 4, as.numeric(.))) %>%
    mutate_all(as.integer)

  return(list("votes_matrix" = mat_votos, "legislator_names" =  names, "party" = legData ))

}


# Crear una polaridad para NOMINATE. Se elige como polos de derecha a UDI o Republicanos. Para la izquierda
# se selecciona al partido comunista o socialista.
# Devueve un vector con dos nombres de diputados, que marcan el polo de izquierda y derecha
create_polarity <- function(data, seed = 123) {

  data <- data.frame(names = data$legislator_names,
                     party = data$party)

  # left side
  if ( sum(data$party == "PC") >= 1 ) {
    left <- "PC"
  } else {
    left <- "PS"
  }

  # right side
  if ( sum(data$party == "PREP") >= 1 ) {
    right <- "PREP"
  } else {
    right <- "UDI"
  }

  set.seed(seed)

  polarity <- data %>%
    filter(party %in% c(left, right)) %>%
    group_by(party) %>%
    sample_n(1) %>%
    ungroup()

  left <- polarity %>%
    filter(party %in% c("PC", "PS")) %>%
    pull(names)

  right <- polarity %>%
    filter(party %in% c("UDI", "PREP")) %>%
    pull(names)

  return(c(right, left))
}

# Recibe datos de w-nominate y devuelve un gráfico de promedios para algunos partidos
plot_mean_party <- function(data) {
  data$legislators %>%
    select(coord1D, party) %>%
    arrange(coord1D) %>%
    filter(!is.na(coord1D)) %>%
    filter(party %in% c("PCS", "UDI", "RN", "PC", "PS", "PPD", "DC", "RD", "EVOP", "PREP")) %>%
    group_by(party) %>%
    summarise(coord1D = mean(coord1D)) %>%
    ggplot(aes(x =  reorder(party, coord1D ),  y = coord1D, color = party, label = party)) +
    geom_point(size = 10) +
    geom_text(hjust=2, vjust=0) +
    coord_flip() +
    labs(title = "W-NOMINATE para votaciones (media partidos)",
         subtitle = "primera dimensión")

}

# Función que crea nominate para un año determinado
# Recibe una año y devuelve ls primeras dimensiones de w-nominate

create_nominate_input <- function(year, votes_sample = 1) {

  # COnseguir los id de cada votación. Se guardaen un dataframe todos los id de las votaciones
  id_votaciones <- descargar_id_votaciones(year)
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

  militancia_diputados <- get_deputy_parties("all")

  roll_call_data <-  create_vote_data(long_data = clean_data, militancia_diputados = militancia_diputados )


  polarity <- create_polarity(roll_call_data)


  rc <- rollcall(roll_call_data$votes_matrix, yea=c(1), nay=c(0),
                 missing=c(2, 3),notInLegis=4, legis.names=roll_call_data$legislator_names, legis.data=roll_call_data$party, desc="UN Votes" )

  result <- wnominate(rc, polarity=polarity)

  result <- append(result, list(year = year))

  return(result)

}
