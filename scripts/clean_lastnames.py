
import pandas as pd
import re
vascos = pd.read_excel('data/apellidos_vinosos.ods', engine='odf')

alemanes= pd.read_excel('data/apellidos_alemanes.ods', engine='odf')
italianos= pd.read_excel('data/apellidos_italianos.ods', engine='odf' )
ingleses= pd.read_excel('data/apellidos_ingleses.ods', engine='odf')
franceses = pd.read_excel('data/apellidos_franceses.ods', engine='odf')

# Editar alemanes
def editar(string):
    editado = re.sub(r'\d|\(|\)', "", string)
    editado = re.sub(u'\xa0', "", editado )    
    editado = re.sub("á", "a", editado)
    editado = re.sub("é", "e", editado )
    editado = re.sub("í", "i", editado )
    editado = re.sub("ó", "o", editado)
    editado = re.sub("ú", "u", editado ) 
    editado = editado.lower()
    return editado

# Limpiar listados de apellidos
alemanes["lastname2"]= list(map(lambda x:editar(x), alemanes.lastname))
italianos["lastname2"]= list(map(lambda x:editar(x), italianos.lastname))
ingleses["lastname2"]= list(map(lambda x:editar(x), ingleses.lastname))
franceses["lastname2"]= list(map(lambda x:editar(x), franceses.lastname))
vascos["lastname2"] = [ re.sub(r':.+', '', str(s)).lower()  for s in vascos.lastname]



def find_lastname(name, lastname_list): 
    for n in name.lower().split()[-2:]:
        edit_n = re.sub("á", "a", n)
        edit_n = re.sub("é", "e", edit_n )
        edit_n = re.sub("í", "i", edit_n )
        edit_n = re.sub("ó", "o", edit_n )
        edit_n = re.sub("ú", "u", edit_n )
        if edit_n in lastname_list:
            return True
    return False
      
alemanes.to_feather("data/german_lastnames.feather")
italianos.to_feather("data/italian_lastnames.feather")
ingleses.to_feather("data/english_lastnames.feather")
franceses.to_feather("data/french_lastnames.feather")
vascos.to_feather("data/vascos_lastnames.feather")
