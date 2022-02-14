#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


USAhousing = pd.read_csv("USA_Housing.csv")


# In[47]:


USAhousing.head()


# In[48]:


USAhousing.info()


# In[49]:


USAhousing.describe()


# In[50]:


sns.pairplot(USAhousing)


# In[51]:


sns.displot(USAhousing["Price"],kde = True)


# In[52]:


sns.heatmap(USAhousing.corr(),cmap = 'coolwarm')


# In[53]:


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[56]:


from sklearn.linear_model import LinearRegression


# In[57]:


lm = LinearRegression()


# In[58]:


lm.fit(X_train,y_train)


# In[59]:


print(lm.intercept_)


# In[60]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[61]:


predictions = lm.predict(X_test)


# In[74]:


sns.regplot(y_test,predictions,scatter_kws={"color": "blue"}, line_kws={"color": "red"})
sns.set(style = 'whitegrid')


# In[75]:


plt.scatter(y_test,predictions)
sns.set(style = 'whitegrid')


# In[63]:


sns.displot((y_test-predictions),bins=50,kde = True);


# In[64]:


from sklearn import metrics


# In[65]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




