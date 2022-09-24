filtrar_polos <- function(data, polaridad, variable) {
  dato <- data %>% 
    filter(polo == polaridad) %>% 
    pull({{variable}})
  return(dato)
}
