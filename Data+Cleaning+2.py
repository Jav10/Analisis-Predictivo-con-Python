
# coding: utf-8

# In[3]:


import pandas as pd
pd.set_option('display.notebook_repr_html', False)

#Leyendo datos con el método read_csv
data = pd.read_csv('capitulo2/titanic3.csv')


# In[4]:


data


# In[6]:


#Leyendo archivos txt
data = pd.read_csv('capitulo2/Customer Churn Model.txt')


# In[7]:


data


# In[34]:


#Abriendo archivos usando el método open
data = open('capitulo2/Customer Churn Model.txt','r')
cols  = data.readline().strip().split(',')
no_cols=len(cols)


# In[35]:


cols


# In[20]:


no_cols


# In[36]:


#Encontrar el número de registros
counter = 0
main_dict = {}

for col in cols:
    main_dict[col]=[]
    
for line in data:
    values = line.strip().split(',')
    for i in range(len(cols)):
        main_dict[cols[i]].append(values[i])
    counter += 1  
    
print("El conjunto de datos tiene %d filas y %d columnas" % (counter, no_cols))
#El método readline se queda en la última ubicación y si se corre otra vez el counter es 0


# In[38]:


df = pd.DataFrame(main_dict)
df.head()


# In[41]:


#Generamos un archivo con un delimitador '/t'
infile = 'capitulo2/Customer Churn Model.txt'
outfile = 'capitulo2/Tab Customer Churn Model.txt'

with open(infile) as infile1:
    with open(outfile, 'w') as outfile1:
        for line in infile1:
            fields = line.split(',')
            outfile1.write('/t'.join(fields))


# In[42]:


data = pd.read_csv('capitulo2/Tab Customer Churn Model.txt',sep='/t')


# In[43]:


data


# In[60]:


#Leyendo datos de una URL
import csv
import urllib

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv'
response = urllib.request.urlopen(url)
cr = csv.reader(response,'excel')
for rows in cr:
    print(rows)


# In[61]:


archivo = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv')


# In[62]:


archivo


# In[64]:


#Leyendo archivos .xls o .xlsx
#read_excel funciona para ambos el segundo argumento es la hoja que se quiere leer
data = pd.read_excel('capitulo2/titanic3.xlsx','titanic3')
data


# In[65]:


#Escribir en un archivo CSV o EXCEL
#Podemos escribir un Data Frame dentro de un archivo excel o csv
#eldataframe.to_csv()
#eldataframe.to_excel()
df.to_excel('capitulo2/dataframe.xls')


# In[66]:


data = pd.read_csv('capitulo2/titanic3.csv')
data.head()


# In[67]:


#filas y columnas del archivo
data.shape


# In[69]:


#Nombres de las columnas
data.columns.values


# In[72]:


#Descripción de los datos
data.describe()


# In[73]:


#Tipo de cada columna
data.dtypes


# In[75]:


#Valores perdidos en el DataFrame
#Devuelve una serie indicando True en la celda con valores perdidos y False para valores no perdidos
#pd.isnull(data['body'])
#lo contrario podria hacerse con
pd.notnull(data['body'])


# In[77]:


#Numéro de entradas con valores perdidos
pd.isnull(data['body']).values.ravel().sum()


# In[78]:


#Numéro de entradas con valores 
pd.notnull(data['body']).values.ravel().sum()


# In[79]:


#Borrar filas si tiene todos sus valores perdidos
data.dropna(axis=0, how='all')


# In[80]:


#Borrar la fila si tiene algun valor perdido
data.dropna(axis=0, how='any')


# In[83]:


#Reemplazar valores perdidos con algún otro valor
#data.fillna(0)
data.fillna('missing')


# In[84]:


#Reemplazar valores perdidos en una columna en particular
data['body'].fillna(0)


# In[85]:


#Cambiar los valores perdidos por el promedio
data['age']


# In[87]:


data['age'].fillna(data['age'].mean())
#También puede usarse la mediana


# In[89]:


#Hay dos métodos para rellenar valores perdidos
#ffill: reemplza los valores perdidos con el valor anterior no perdido
#data['age'].fillna(method='ffill')
#backfill: reemplza los valores perdidos con el valor siguiente no perdido
#data['age'].fillna(method='backfill')


# In[90]:


#Creando variables ficticias
#Es un método para crear variables separadas para cada categoria de un variable categorica
dummy_sex = pd.get_dummies(data['sex'], prefix='sex')
dummy_sex


# In[95]:


#Agregando las nuevas variables
column_name = data.columns.values.tolist() # convirtiendo los valores en una lista
column_name.remove('sex') #Removemos la variable categórica
data[column_name].join(dummy_sex) #Agregamos la variable ficticia


# In[96]:


#Visualizando conjuntos de datos con gráficas básicas
data = pd.read_csv('capitulo2/Customer Churn Model.txt')


# In[99]:


get_ipython().magic('matplotlib inline')
#Gráfica de dispersión
data.plot(kind='scatter',x='Day Mins', y='Day Charge')


# In[98]:


import matplotlib.pyplot as plt
figure, axs = plt.subplots(2,2,sharey=True,sharex=True)
data.plot(kind='scatter',x='Day Mins',y='Day Charge',ax=axs[0][0])
data.plot(kind='scatter',x='Night Mins',y='Night Charge',ax=axs[0][1])
data.plot(kind='scatter',x='Day Calls',y='Day Charge',ax=axs[1][0])
data.plot(kind='scatter',x='Night Calls',y='Night Charge',ax=axs[1][1])


# In[100]:


#Histogramas
plt.hist(data['Day Calls'],bins=8)
plt.xlabel('Day Calls Value')
plt.ylabel('Frequency')
plt.title('Frequency of Day Calls')


# In[101]:


#Gráfica de caja y bigote
plt.boxplot(data['Day Calls'])
plt.ylabel('Day Calls')
plt.title('Box Plot of Day Calls')


# In[102]:


'''
La caja azul es de primordial importancia. El borde inferior horizontal de la caja especifica el primer cuartil,
mientras que el borde superior horizontal especifica el tercer cuartil. 
La línea horizontal en rojo especifica el valor mediano. La diferencia en los valores de cuartil primero y 
tercero se denomina Rango intercuartílico o IQR. Los bordes horizontales inferior y 
superior en negro especifican los valores mínimo y máximo, respectivamente.
'''

