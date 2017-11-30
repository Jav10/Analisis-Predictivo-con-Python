
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

datos = {'Mes':[1,2,3,4,5,6],'Ventas':[7000,9000,5000,11000,10000,13000]}
df = pd.DataFrame(datos)
df


# In[4]:


pendiente = (np.sum(df['Mes'])*np.sum(df['Ventas'])-len(df)*np.sum(df['Mes']*df['Ventas']))/((np.sum(df['Mes']))**2-len(df)*np.sum(df['Mes']*df['Mes']))
pendiente


# In[5]:


intercepto = np.mean(df['Ventas'])-pendiente*np.mean(df['Mes'])
intercepto


# In[11]:


#Coeficiente de correlación
R =  np.cov(np.array(df['Mes'],df['Ventas']))/(np.std(df['Mes'])*np.std(df['Ventas']))
R


# In[7]:


#Para el mes 7
MesSiete = intercepto + pendiente * 7
MesSiete


# In[10]:


#Coeficiente de determinación
(R**2)


# In[16]:


import matplotlib.pyplot as plt
def funcion(x):
    global pendiente
    global intercepto
    return intercepto + pendiente*x

plt.plot(df['Mes'],df['Ventas'])
plt.scatter(df['Mes'],df['Ventas'])
plt.plot(df['Mes'], [funcion(i) for i in df['Mes']])
plt.show()

