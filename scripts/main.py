import os  


mode = "word"

# Generación de cada uno de los polos
string = "python /home/klaus/discursos_politicos/scripts/create_poles_vectors.py" + " " + mode
os.system(string) 

# Convertir texto en vectores
string = "python /home/klaus/discursos_politicos/scripts/convert_text_to_vec.py" + " " + mode
os.system(string) 

# Evaluar el rendimiento
string = "python /home/klaus/discursos_politicos/scripts/evaluate_poles_vectors.py" + " " + mode
os.system(string) 


