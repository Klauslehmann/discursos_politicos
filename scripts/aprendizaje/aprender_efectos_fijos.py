

import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from linearmodels import RandomEffects
from linearmodels.datasets import jobtraining

import statsmodels.api as sm
import statsmodels.formula.api as smf


data = jobtraining.load()
year = pd.Categorical(data.year)

# Generar un identificador para cada observación. En este caso se usa el id de la empresa y un año
data = data.set_index(["fcode", "year"])
data["year"] = year

# Definir las variables independientes. Se agrega un intercepto al modelo
exog_vars = ["grant", "employ"]
exog = sm.add_constant(data[exog_vars])

# Estimar modelo de efectos aleatorios
mod = RandomEffects(data.clscrap, exog)
re_res = mod.fit()
print(re_res)

# Modelo de efectos fijos
mod = PanelOLS(data.clscrap, exog, entity_effects=True) # entity_effects=True
re_res = mod.fit()
print(re_res)


# Efectos fijos con el enfoque dummy y estilo python
data = jobtraining.load()
data["year"] = pd.Categorical(data.year)
data["dummy_firm"] = pd.Categorical(data.fcode)

dummy = pd.get_dummies(data.fcode)

new_data = pd.concat([data, dummy ], axis=1)   
variables = ["grant", "employ"]
exog_vars =     list(dummy.columns) + variables

exog = sm.add_constant(new_data[exog_vars])
FE_ols1 = sm.OLS( new_data.clscrap, exog, missing='drop' ).fit()
print(FE_ols1.summary())


# Efectos fijos, con el enfoque dummy y usando estilo fórmula de R
data = jobtraining.load()
FE_ols2 = smf.ols(formula="clscrap ~  grant + employ + C(fcode)", data = data,
                 missing='drop' 
                 ).fit()
print(FE_ols2.summary())





import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

dict = {'industry': ['mining', 'transportation', 'hospitality', 'finance', 'entertainment'],
  'debt_ratio':np.random.randn(5), 'cash_flow':np.random.randn(5) + 90} 

df = pd.DataFrame.from_dict(dict)

x = df[['debt_ratio', 'industry']]
y = df['cash_flow']

# NB. unlike sm.OLS, there is "intercept" term is included here
x = smf.ols(formula="cash_flow ~ debt_ratio + C(industry)", data=df).fit()
print(x.summary())
