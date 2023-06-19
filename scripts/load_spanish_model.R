
# Crear un directorio para almacenar el modelo
dir.create("model", showWarnings = F)

# Descargar el modelo, si no está disponible en la carpeta model
if (!file.exists("model/spanish-gsd-ud-2.5-191206.udpipe") ) {
  udpipe_download_model(model_dir = "model", language = "spanish")
}

# Cargar el modelo para usarlo después
model <- udpipe_load_model(file = "model/spanish-gsd-ud-2.5-191206.udpipe")

