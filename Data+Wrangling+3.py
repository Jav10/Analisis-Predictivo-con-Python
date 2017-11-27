
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv('capitulo2/Customer Churn Model.txt')


# In[2]:


#Seleccionando columnas
account_length = data['Account Length']
account_length.head()


# In[3]:


type(account_length)


# In[4]:


#Seleccionando múltiples columnas
subdata = data[['Account Length','VMail Message','Day Calls']]
subdata.head()


# In[5]:


type(subdata)


# In[8]:


#No seleccionar algunas columnas del data frame
wanted = ['Account Length','VMail Message','Day Calls']
column_list =  data.columns.values.tolist()
sublist = [x for x in column_list if x not in wanted]
subdata = data[sublist]
subdata.head()


# In[9]:


#Seleccionar filas
data[1:50]


# In[11]:


data[25:75]


# In[12]:


data[:50]


# In[6]:


#Selección booleana
data1 = data[data['Night Mins']>200]
data1.shape


# In[7]:


data1 = data[data['State']=='VA']
data1.shape


# In[13]:


data1 = data[(data['Night Mins']>150) & (data['State']=='VA')]
data1.shape


# In[14]:


data1 = data[(data['Night Mins']>150) | (data['State']=='VA')]
data1.shape


# In[18]:


subdata_first_50 = data[['Account Length','VMail Message','Day Calls']][1:51]
subdata_first_50


# In[19]:


#Método ix
data.ix[1:100,1:6]


# In[20]:


#Método iloc y loc
data.iloc[:,1:6]


# In[21]:


data.iloc[1:100,:]


# In[22]:


data.iloc[1:100,[2,5,7]]


# In[23]:


data.iloc[[1,2,5],[2,5,7]]


# In[24]:


data.loc[[1,2,5],['Area Code','VMail Plan','Day Mins']]


# In[26]:


#Creando columnas nuevas
data['Total Mins'] = data['Day Mins'] + data['Eve Mins'] + data['Night Mins']
data['Total Mins'].head()


# In[27]:


data.columns.values


# In[28]:


#Generandos números aleatorios y sus usos
import numpy as np
#Genera números enteros entre a y b
np.random.randint(1,100)


# In[29]:


#Números entre 0 y 1
np.random.random()


# In[30]:


#función para generar n números entre a y b
def randint_range(n,a,b):
    x=[]
    for i in range(n):
        x.append(np.random.randint(a,b))
    return x    

y = randint_range(10,2,1000)
y


# In[31]:


#Generar números enteros entre 0 y 100, los cuales son todos múltiplos de 5
import random
for i in range(3):
    print(random.randrange(0,100,5))


# In[42]:


#Ordenar una lista o array de forma aleatoria
a = np.array(range(10))
np.random.shuffle(a)
a


# In[45]:


#Seleccionar un elemento aleatorio de una lista dada
column_list = data.columns.values.tolist()
np.random.choice(column_list)


# In[46]:


#Sembrar un número aleatorio - repitiendo los números aleatorios
np.random.seed(1)
for i in range(5):
    print(np.random.random())


# In[49]:


for i in range(5):
    print(np.random.random())


# In[51]:


#Generando números aleatorios siguiendo distribuciones de probabilidad
#Distribución uniforme
randnum = np.random.uniform(1,100,100)
randnum


# In[59]:


#Graficando la distribución
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
a = np.random.uniform(1,100,100)
b = range(1,101)
plt.hist(a)


# In[62]:


#Los datos usados son pocos, probaremos con 1,000,000
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
a = np.random.uniform(1,100,1000000)
plt.hist(a)


# In[64]:


#Distribución Normal
#La función random.randn genera valores con distribución normal
a = np.random.randn(100)
a


# In[65]:


a = np.random.randn(100)
b=range(1,101)
plt.plot(b,a)


# In[67]:


#Podriamos generar una matriz donde sus elementos sigan una disribución normal
#Si no se específica un valor se genera un solo elemento
a = np.random.randn(2,4)
a


# In[68]:


#Para generar un distribución con media y desviación estandar diferentes de 0 y 1 como por ejemplo 1.5 y 2.5 respectivamente
#dado que S=(X-u)/s  
a = 2.5*np.random.randn(100)+1.5
a


# In[69]:


a = np.random.randn(100000)
b = range(1,101)
plt.hist(a)


# In[74]:


#Usando la simulación Monte-Carlo para encontrar el valor de PI
pi_avg = 0
pi_value_list = []
for i in range(100):
    value = 0
    x = np.random.uniform(0,1,1000).tolist()
    y = np.random.uniform(0,1,1000).tolist()
    for j in range(1000):
        z = np.sqrt(x[j]*x[j]+y[j]*y[j])
        if z<=1:
            value+=1
    float_value = float(value)
    pi_value = float_value*4/1000
    pi_value_list.append(pi_value)
    pi_avg += pi_value

pi = pi_avg/100
print(pi)
ind = range(1,101)
fig = plt.plot(ind, pi_value_list)
fig
            


# In[75]:


#Generando un Data Frame ficticio
d = pd.DataFrame({'A':np.random.randn(10), 'B':2.5*np.random.randn(10)+1.5})
d


# In[76]:


#Las variables categoricas pueden ser pasadas como una lista para ser parte de un DataFrame ficticio
data = pd.read_csv('capitulo2/Customer Churn Model.txt')
column_list = data.columns.values.tolist()
a = len(column_list)
d = pd.DataFrame({'Column_Name': column_list, 'A':np.random.randn(a),'B':2.5*np.random.randn(a)+1.5})
d


# In[80]:


#El índice también puede ser pasado como parámetro para esta función
d = pd.DataFrame({'A':np.random.randn(10), 'B':2.5*np.random.randn(10)+1.5}, index=range(10,20))
d


# In[93]:


#Agrupando los datos - Agregación, filtración y transformación
#Agregando datos sobre variables categóricas
import pandas as pd
import numpy as np

a = ['Male','Female']
b = ['Rich','Poor','Middle Class']
gender = []
seb = []
for i in range(1,101):
    gender.append(np.random.choice(a))
    seb.append(np.random.choice(b))
height = 30*np.random.randn(100)+155
weight = 20*np.random.randn(100)+60
age= 10*np.random.randn(100)+35
income = 1500*np.random.randn(100)+15000
df = pd.DataFrame({"Gender":gender, "Height":height, "Weight":weight,"Age":age,"Income":income,"Socio-Eco":seb})
df.head()


# In[95]:


df.shape


# In[96]:


#Agrupando sobre variables categóricas
df.groupby('Gender')


# In[97]:


grouped = df.groupby('Gender')
grouped.groups


# In[98]:


grouped = df.groupby('Gender')
for names, groups in grouped:
    print(names)
    print(groups)


# In[99]:


#Seleccionar un solo grupo
grouped.get_group('Female')


# In[102]:


#Podemos agrupar sobre más de una variable categórica
grouped = df.groupby(['Gender','Socio-Eco'])
for names, groups in grouped:
    print(names)
    print(groups)


# In[103]:


#Agregación
#La agregación practicamente significa aplicar una función a todos los grupos a la vez y obtener un resultado de ese grupo
#funciones: sum, mean, describe, size, etc.
grouped = df.groupby(['Gender','Socio-Eco'])
grouped.sum()


# In[104]:


#Calcular el tamaño de cada grupo
grouped = df.groupby(['Gender','Socio-Eco'])
grouped.size()


# In[105]:


#Podemos usar la función describe para obtener un resumen esta´distico de cada grupo por separado
grouped = df.groupby(['Gender', 'Socio-Eco'])
grouped.describe()


# In[111]:


#Seleccionar columnas del grupo
grouped_income = grouped['Income']
for i,j in grouped_income:
    print(i)
    print(j)
    


# In[112]:


#Podemos aplicar diferentes funciones a diferentes columnas
grouped.aggregate({'Income':np.sum,'Age':np.mean, 'Height':np.std})


# In[113]:


#Podemos usar funciones lambda para ver la proporcion de la media y la desviación estandar
grouped.aggregate({'Age':np.mean,'Height':lambda x:np.mean(x)/np.std(x)})


# In[114]:


#Podemos aplicar varias funciones a todas las columnas al mismo tiempo
grouped.aggregate([np.sum,np.mean,np.std])


# In[118]:


#Filtración
#Podemos filtrar elementos basados en las propiedades del grupo
grouped['Age'].filter(lambda x: x.sum()>20)


# In[119]:


#Transformación
zscore = lambda x: (x - x.mean()) / x.std()
grouped.transform(zscore)


# In[120]:


#Podriamos usar la transformación para rellenar los valores perdidos con la media
#f = lambda x: x.fillna(x.mean())
#grouped.transform(f)


# In[121]:


#Operaciones diversas
#Podemos seleccionar la primera fila de cada grupo
grouped.head(1)


# In[122]:


#Así lo hariamos para ver las últimas filas de cada grupo
grouped.tail(1)


# In[123]:


#Podemos usar la función nth para seleccionar la enésima fila de un grupo
grouped.nth(1)


# In[125]:


#Ordenar los datos antes de agrupar
df1 = df.sort_values(['Age', 'Income'])
grouped = df1.groupby('Gender')
grouped.head(1)


# In[126]:


#Muestras Aleatorias - dividir un dataset en entrenamiento y testeo
#Método 1 - Dividir los datos usando la distribución normal estandar

data = pd.read_csv('capitulo2/Customer Churn Model.txt')
len(data)


# In[128]:


#Generamos números aleatorios y creamos un filtro para particionar los datos
a = np.random.randn(len(data))
check = a<0.8
training = data[check]
testing = data[~check]


# In[129]:


len(training)


# In[132]:


len(testing)


# In[134]:


#Método 2 - usando sklearn

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.2)
len(test)/3333


# In[149]:


#Método 3 - usando la función shuffle
with open('capitulo2/Customer Churn Model.txt', 'r') as f:
    data = f.read().split('\n')
np.random.shuffle(data)
train = 3*len(data)/4
test = len(data)/4
train_data = data[:int(train)]
test_data = data[int(train)+1:]
print(len(train_data), len(test_data))


# In[150]:


#Concatenando y Anexando datos
data1 = pd.read_csv('capitulo3/winequality-red.csv',sep=';')
data1.head()


# In[151]:


data1.columns.values


# In[152]:


data1.shape


# In[153]:


data2 = pd.read_csv('capitulo3/winequality-white.csv', sep=';')
data2.head()


# In[154]:


data2.columns.values


# In[155]:


data2.shape


# In[157]:


wine_total = pd.concat([data1,data2],axis=0)
wine_total.shape


# In[158]:


wine_total.head()

