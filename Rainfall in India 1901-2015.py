#!/usr/bin/env python
# coding: utf-8

# # Rainfall Prediction in India 1901-2015

# In[1]:


import lazypredict


# In[2]:


import pandas as pd
import bamboolib as bam
bam.enable()
data = pd.read_csv(r'C:\Users\HP\Downloads\rainfall in india 1901-2015.csv')
data


# In[3]:


Value_Mapping = {'ANDAMAN & NICOBAR ISLANDS' : 1, 'ARUNACHAL PRADESH' : 2,
       'ASSAM & MEGHALAYA' :3, 'NAGA MANI MIZO TRIPURA' :4,
       'SUB HIMALAYAN WEST BENGAL & SIKKIM' : 5, 'GANGETIC WEST BENGAL' : 6,
       'ORISSA' : 7, 'JHARKHAND' : 8, 'BIHAR' : 9, 'EAST UTTAR PRADESH' : 10,
       'WEST UTTAR PRADESH' : 11, 'UTTARAKHAND' : 12, 'HARYANA DELHI & CHANDIGARH' : 13,
       'PUNJAB' : 14, 'HIMACHAL PRADESH' : 15, 'JAMMU & KASHMIR' : 16, 'WEST RAJASTHAN': 17,
       'EAST RAJASTHAN' : 18, 'WEST MADHYA PRADESH' : 19, 'EAST MADHYA PRADESH' : 20,
       'GUJARAT REGION' : 21, 'SAURASHTRA & KUTCH' : 22, 'KONKAN & GOA' : 23,
       'MADHYA MAHARASHTRA' : 24, 'MATATHWADA' : 25, 'VIDARBHA' : 26, 'CHHATTISGARH' : 27,
       'COASTAL ANDHRA PRADESH' : 28, 'TELANGANA' : 29, 'RAYALSEEMA' : 30, 'TAMIL NADU' : 31,
       'COASTAL KARNATAKA' : 32, 'NORTH INTERIOR KARNATAKA' :33,
       'SOUTH INTERIOR KARNATAKA' : 34, 'KERALA' : 35, 'LAKSHADWEEP' : 36}
data['STATES'] = data['SUBDIVISION'].map(Value_Mapping)
df = data.drop('SUBDIVISION',axis = 1)
df


# In[4]:


df=df.fillna(df.median())
df


# In[5]:


df.isnull().sum()


# In[ ]:


from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

y = df['ANNUAL']
X = df.drop("ANNUAL",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)


# In[ ]:


from lazypredict.Supervised import LazyRegressor

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)


# In[ ]:




