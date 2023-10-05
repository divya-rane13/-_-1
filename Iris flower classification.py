#!/usr/bin/env python
# coding: utf-8

# # Importing neccessary python libraries

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# # Importing data
# 

# In[22]:


iris=pd.read_csv("Iris.csv")
iris.head()


# In[23]:


iris=iris.drop("Id",axis=1)
iris


# In[24]:


iris.describe()


# In[25]:


print("Target",iris["Species"].unique())


# In[26]:


import plotly.express as px
figure=px.scatter(iris,x="SepalWidthCm",y="SepalLengthCm",color="Species")
figure.show()


# # Iris Classification Model

# In[27]:


x = iris.drop("Species", axis=1)
y = iris["Species"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)


# # Give input for to predict the iris flower species
# 

# In[20]:


new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(new)
print("Prediction: {}".format(prediction))


# In[ ]:




