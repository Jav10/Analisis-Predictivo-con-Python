import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

advert = pd.read_csv('Advertising.csv')
print(advert.head())

model1 = smf.ols(formula='Sales~TV', data=advert).fit()
print(model1.params) #valores de a y b

#P-values
print(model1.pvalues)

#R^2
print(model1.rsquared)

#Resumen
print(model1.summary())

#Predecir los valores de ventas
sales_pred = model1.predict(pd.DataFrame(advert['TV']))
print(sales_pred)

advert.plot(kind='scatter',x='TV', y='Sales')
plt.plot(pd.DataFrame(advert['TV']), sales_pred, c='red', linewidth=2)
plt.show()

#Calculado el RSE

advert['sales_pred'] = 0.047537*advert['TV']+7.03
advert['RSE'] = (advert['Sales']-advert['sales_pred'])**2
RSEd = advert.sum()['RSE']
RSE = np.sqrt(RSEd/198)
salesmean = np.mean(advert['Sales'])
error = RSE/salesmean
print(RSE,salesmean, error*100)
