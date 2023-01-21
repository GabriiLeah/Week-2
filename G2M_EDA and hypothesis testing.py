#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)

warnings.filterwarnings("ignore")

#Loading Cab_df file into the system
df=pd.read_csv("C:/Users/Leah Estioco/Documents/dataglacierinternship - 2023/Week 2/Cab_Data.csv")
df.head(10)


# In[2]:


df.info()


# In[76]:


df.dtypes


# In[4]:


df.describe(include = 'all', datetime_is_numeric=True)


# In[5]:


#Sorting data by company
x=df["Company"].to_numpy()
df["Company"].unique()


# In[6]:


#Sorting based om cities
x=df["City"].to_numpy()
df["City"].unique()


# In[7]:


#Loading City.csv file into the system
city=pd.read_csv("C:/Users/Leah Estioco/Documents/dataglacierinternship - 2023/Week 2/City.csv")
city.head(10)


# In[8]:


k=city["Users"].to_numpy()
city["Users"].unique()


# In[9]:


city.info()


# In[10]:


city.dtypes


# In[11]:


city.describe(include = 'all', datetime_is_numeric=True)


# In[12]:


#Loading Transaction_ID.csv file into the system
transaction=pd.read_csv("C:/Users/Leah Estioco/Documents/dataglacierinternship - 2023/Week 2/Transaction_ID.csv")
transaction.head(10)


# In[13]:


transaction.info()


# In[14]:


transaction.dtypes


# In[15]:


transaction.describe(include = 'all', datetime_is_numeric=True)


# In[16]:


#Loading Customer_ID.csv file into the system
customer=pd.read_csv("C:/Users/Leah Estioco/Documents/dataglacierinternship - 2023/Week 2/Customer_ID.csv")
customer.head(10)


# In[17]:


transaction.info()


# In[18]:


customer.describe(include = 'all', datetime_is_numeric=True)


# In[19]:


#Merging the whole datasets

fin=df.merge(transaction, on="Transaction ID").merge(customer, on="Customer ID").merge(city,on="City")
fin.head(5)


# In[20]:


sns.pairplot(fin.head(1000), hue = 'Company',palette="hls")


# In[21]:


data_corr = fin.corr()
sns.heatmap(data_corr, cmap="crest",annot=data_corr.rank(axis="columns"),linewidth=.5)


# In[22]:


# Define the figure size
plt.figure(figsize = (16, 9))

# Cutomize the annot
annot_kws={'fontsize':10,                      # To change the size of the font
           'fontstyle':'italic',               # To change the style of font 
           'fontfamily': 'serif',              # To change the family of font 
           'alpha':1 }                         # To change the transparency of the text  


# Customize the cbar
cbar_kws = {"shrink":1,                        # To change the size of the color bar
            'extend':'min',                    # To change the end of the color bar like pointed
            'extendfrac':0.1,                  # To adjust the extension of the color bar
            "drawedges":True,                  
           }

# take upper correlation matrix
matrix = np.triu(data_corr)

# Generate heatmap correlation
ax = sns.heatmap(data_corr, mask = matrix, cmap = 'crest', annot = True, linewidth = 1.5 ,annot_kws= annot_kws, cbar_kws=cbar_kws)

# Set the title etc
plt.title('Correlation Heatmap of "G2M Insight for Cab Investment", fontsize = 20')

# Set the size of text
sns.set(font_scale = 1.2)


# In[23]:


user=fin.groupby('Company')
avg_user = user.Users.mean()
index = avg_user.index
value = avg_user.values 
colors=['lightgrey','lightblue']


# In[24]:


plt.pie(avg_user,labels=index,colors=colors,
autopct='%1.1f%%', shadow=False, startangle=120)
plt.axis('equal')
plt.title('Users Travel', fontsize = 15)
plt.show()


# In[92]:


sns.set(style = 'whitegrid')

plt.figure(figsize = (16, 9))
sns.boxplot(fin['Company'], fin['Price Charged'],color='grey')
plt.title('Price Charged of Both Companies', fontsize=20)
plt.show()


# In[60]:


plt.figure(figsize = (16, 9))
plt.hist(fin['KM Travelled'], bins = 40,color='lightblue')
plt.title('Km Travelled Distribution', fontsize=20)
plt.ylabel('Frequency')
plt.xlabel('Km Travelled')
plt.show()


# In[57]:


plt.figure(figsize = (16, 9))

ax = sns.countplot(x="Company", hue="Payment_Mode", data=fin,color='lightblue')
plt.title('Payment Mode in both companies', fontsize=25)
plt.show()


# In[28]:


gcab=fin.groupby(['Company','Gender'])
gcab = gcab['Customer ID'].nunique()
print(gcab)


# In[29]:


co = ['lavender','lightblue','royalblue','lightgrey']
labs = gcab.index
vals = gcab.values
figp, axp = plt.subplots(figsize=(10,7))
axp.pie(vals , labels=labs, autopct='%1.1f%%',shadow=False, startangle=90,colors=co)
axp.axis('equal')

plt.title('Customer share per gender per cab', fontsize = 15)
plt.show()


# In[64]:


city_users = fin.groupby('City')
city_users = city_users.Users.count()
labs = city_users.index
vals = city_users.values

fig, ax = plt.subplots(figsize =(16, 9))
 
# Horizontal Bar Plot
ax.barh(labs, vals,color='lightblue')

for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)

ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)

ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)

ax.invert_yaxis()

for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y()+0.5,
             str(round((i.get_width()), 2)),
             fontsize = 12, fontweight ='bold',
             color ='grey')
    
ax.set_title('Users per City',
             loc ='Center',fontsize=20 )

fig.text(0.9, 0.15, 'Jeeteshgavande30', fontsize = 15,
         color ='grey', ha ='right', va ='bottom',
         alpha = 0.7)

plt.show()


# In[65]:


company = fin.groupby('Company')
price_charged = company['Price Charged'].mean()
cost_trip = company['Cost of Trip'].mean()
c = cost_trip.index
c_v = cost_trip.values
c_p = price_charged.values


# In[68]:


plt.style.use('fivethirtyeight')
plt.figure(figsize = (16, 9))
plt.bar(c, c_p, edgecolor='black', label="Revenue",color='grey')
plt.bar(c, c_v, edgecolor='black', label="Profit",color='lightblue')
plt.title('Profit Margin')
plt.ylabel('Price Charged')
plt.xlabel('Cost of Trip')
plt.legend()
plt.show()


# In[72]:


plt.figure(figsize = (16, 9))

sns.scatterplot(data=fin, x="KM Travelled", y='Price Charged', hue='Company')
plt.title('Price Charged w.r.t Distance',fontsize = 20)
plt.ylabel('Price Charged',fontsize = 14)
plt.xlabel('KM Travelled',fontsize = 14)
plt.show()


# In[81]:


urp = (city['Users'] /city['Population']) * 100 
city = city['City']


# In[82]:


# Get the list of color
from random import randint

colors = []
n = 16

for i in range(n):
    colors.append('#%06X' % randint(0, 0xFFFFFF))


# In[83]:


plt.figure(figsize = (16, 9))
plt.bar(city, urp, edgecolor='black', color = colors)
plt.gcf().autofmt_xdate()
plt.title('Users Respective Population')
plt.ylabel('Percentage (%)')
plt.xlabel('Cities')
plt.show()


# In[94]:


sns.set(style = 'darkgrid') 

plt.figure(figsize = (16, 9))

sns.violinplot(fin['Gender'], fin['Age'], hue = fin['Company'], palette = 'PiYG', inner = 'quartiles')
plt.title('Avg age of users', fontsize=20)
    
plt.show()


# In[91]:


sns.set(style = 'darkgrid')

plt.figure(figsize = (16, 9))

sns.boxplot(fin['Company'], fin['Income (USD/Month)'],color='lightblue')
plt.title('User Income', fontsize=20)
plt.show()


# In[95]:


a = fin[(fin.Gender=='Male')&(fin.Company=='Pink Cab')].groupby('Transaction ID').Profit.mean()
b = fin[(fin.Gender=='Female')&(fin.Company=='Pink Cab')].groupby('Transaction ID').Profit.mean()
print(a.shape[0],b.shape[0])

_, p_value = stats.ttest_ind(a.values,
                             b.values,
                             equal_var=True)

print('P value is ', p_value)

if(p_value<0.05):
    print('We accept alternative hypothesis (H1) that there is a difference regarding gender for Pink Cab')
else:
    print('We accept null hypothesis (H0) that there is no difference regarding gender for Pink Cab')


# In[97]:


a = fin[(fin.Gender=='Male')&(fin.Company=='Yellow Cab')].groupby('Transaction ID').Profit.mean()
b = fin[(fin.Gender=='Female')&(fin.Company=='Yellow Cab')].groupby('Transaction ID').Profit.mean()
print(a.shape[0],b.shape[0])

_, p_value = stats.ttest_ind(a.values,
                             b.values,
                             equal_var=True)

print('P value is ', p_value)

if(p_value<0.05):
    print('We accept alternative hypothesis (H1) that there is a difference regarding gender for Yellow Cab')
else:
    print('We accept null hypothesis (H0) that there is no difference regarding gender for Yellow Cab')


# In[ ]:




