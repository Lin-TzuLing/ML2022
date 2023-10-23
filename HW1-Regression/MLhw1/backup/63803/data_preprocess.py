#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
from dateutil import relativedelta
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
import argparse


# ## argparse 

# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('data_path',type=str, help='設定data來源路徑')
parser.add_argument('corr_fig_path', type=str, help='設定corr fig儲存路徑')
parser.add_argument('processed_data_path', type=str, help='設定processed data儲存路徑')
args = parser.parse_args()


# ## read data

# In[2]:

DATA_PATH = args.data_path
train = pd.read_csv(DATA_PATH+"/train-v3.csv")
valid = pd.read_csv(DATA_PATH+"/valid-v3.csv")
test = pd.read_csv(DATA_PATH+"/test-v3.csv")
print('data num => train: {}, valid: {}, test: {}'.format(len(train),len(valid),len(test)))


# In[3]:


train.sort_values("id").head(5)


# ## check train data info

# In[4]:


train.info()


# In[5]:


train.describe(include=['int64','float64'])


# ## price distribution

# In[6]:


sns.displot(train['price'],height=2, aspect=1)


# In[7]:


sns.displot(valid['price'],height=2, aspect=1)


# ## log(price) to get normal distribution

# In[8]:


train['price'].apply(lambda x: np.log1p(x)).hist(bins=30)


# In[9]:


train['price'] = train['price'].apply(lambda x: np.log1p(x))
valid['price'] = valid['price'].apply(lambda x: np.log1p(x))


# ## turn renovated as 0/1 

# In[10]:


print((train['yr_renovated'] != 0).sum())
print((valid['yr_renovated'] != 0).sum())
print((test['yr_renovated'] != 0).sum())


# In[11]:


def renovate(df): 
    if 'yr_renovated' in df.columns:
        df['renovated'] = df['yr_renovated'].astype(bool).astype(int)
        df = df.drop(['yr_renovated'],axis=1)
    return df

train = renovate(train)
valid = renovate(valid)
test = renovate(test)

train.describe(include=['int32'])
train['renovated'].corr(train['price'])


# ## transfer lat & long 

# In[12]:


def latlong_dist(long1, lat1, long2, lat2):
    # radians() for long1, lat1, long2, lat2     
    long1, lat1, long2, lat2 = map(math.radians, [long1, lat1, long2, lat2]) 
    diff_long = long2 - long1 
    diff_lat = lat2 - lat1 
    a = math.sin(diff_lat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(diff_long/2)**2 
    c = 2 * math.asin(math.sqrt(a)) 
    # const r     
    r = 6371 
    return c * r


# In[13]:


# mean long and lat
# FIXED_LONG = -122.214565
# FIXED_LAT = 47.558910

def longlat_transform(df):
    FIXED_LONG = df['long'].mean()
    FIXED_LAT = df['lat'].mean()
    df['distance'] = df.apply(lambda row: latlong_dist(FIXED_LONG, FIXED_LAT, row['long'], row['lat']), axis=1)
    df['greater_long'] = (df['long'] >= FIXED_LONG).astype(int)
    df['less_long'] = (df['long'] < FIXED_LONG).astype(int)
    df['greater_lat'] = (df['lat'] >= FIXED_LAT).astype(int)
    df['less_lat'] = (df['lat'] < FIXED_LAT).astype(int)
    return df

train = longlat_transform(train)
valid = longlat_transform(valid)
test = longlat_transform(test)


# In[14]:


train.head(2)


# ## calculate date duration between build-to-sale

# In[15]:


# 整合sale_yr, sale_month, sale_day，改成sale_date(Datetime)
def combine_saleDate(df):
    if set(['sale_yr', 'sale_month', 'sale_day']).issubset(df.columns):
        df['sale_date'] = datetime.date.today()
        for idx, row in df.iterrows():
            saleDate_str = (str(row['sale_yr'])+'-'+str(row['sale_month'])+'-'+str(row['sale_day']))
            saleDate = datetime.datetime.strptime(saleDate_str, '%Y-%m-%d')
            df.at[idx,'sale_date'] = saleDate
        df = df.drop(['sale_yr', 'sale_month', 'sale_day'], axis=1)
    return df

# 算建造日期到銷售日期的時間，built-to-sale(int)
def build_to_sale(df):   
    if 'yr_built' in df.columns:
        df['built2sale_day'], df['built2sale_year'] = 0, 0
        for idx, row in df.iterrows():
            builtDate_str = (str(row['yr_built'])+'-06-15')
            builtDate = datetime.datetime.strptime(builtDate_str, '%Y-%m-%d')
            df.at[idx,'built2sale_day'] = (row['sale_date']-builtDate).days
            df.at[idx,'built2sale_year'] = relativedelta.relativedelta(row['sale_date'], builtDate).years
        df = df.drop(['yr_built','sale_date'], axis=1)
    return df


train = build_to_sale(combine_saleDate(train))
valid = build_to_sale(combine_saleDate(valid))
test = build_to_sale(combine_saleDate(test))

train.head(5)


# ## correlation ( feature/price )

# In[16]:


f, ax = plt.subplots(figsize=(15,12))
sns.heatmap(train.corr(), vmax=0.8, square=True)


# In[17]:

CORR_FIG_PATH = args.corr_fig_path
col_list = [e for e in list(train.columns) if e not in ('id','price')]
highCorr_feature = []
for col in col_list:
    corr = train[col].corr(train['price'])
    print('{} : {}'.format(col,corr))
    if abs(corr)>0.00:
        highCorr_feature.append(col)
    # plot correlation     
    plot = sns.scatterplot(x=col, y="price", data=train).get_figure()
    plot.savefig(CORR_FIG_PATH+'/'+str(col)+'_corr='+str(corr)[:6]+'.png')
    plot.get_figure().clf()


# In[18]:


train.corr().loc['price'].sort_values(ascending=False)


# for col in col_list:
#     # plot distribution     
#     plot = sns.displot(train[col])
#     plt.savefig('./dist_fig/'+col+'.png')
#     plt.clf()

# In[19]:


train = train[['id','price','zipcode']+highCorr_feature]
valid = valid[['id','price','zipcode']+highCorr_feature]
test = test[['id','zipcode']+highCorr_feature]

train.head(3)


# ## normalize high_Corrfeature

# In[20]:


train.columns


# In[21]:


norm_col = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'renovated', 'distance', 'built2sale_day',
       'built2sale_year']
normalize = StandardScaler().fit_transform

def norm_df(df):
    normalize_df = df.copy()
    normalize_df[norm_col] = normalize(df[norm_col])
    return normalize_df

normalized_train = norm_df(train)
normalized_valid = norm_df(valid)
normalized_test = norm_df(test)


# In[22]:


normalized_train.describe()


# In[23]:


normalized_train.head(3)


# ## zip code get_dummy 

# In[28]:


def zip_dummy(df):
    if 'zipcode' in df.columns:
        df = pd.get_dummies(df,columns=['zipcode'],prefix='zip_')
#         df = df.drop('zipcode', axis=1)
    return df


# In[29]:


normalized_train = zip_dummy(normalized_train)
normalized_valid = zip_dummy(normalized_valid)
normalized_test = zip_dummy(normalized_test)


# In[33]:


normalized_train.head(2)


# In[34]:

PROCESSED_DATA_PATH = args.processed_data_path
normalized_train.to_csv(PROCESSED_DATA_PATH+'/train.csv', index=False)
normalized_valid.to_csv(PROCESSED_DATA_PATH+'/valid.csv', index=False)
normalized_test.to_csv(PROCESSED_DATA_PATH+'/test.csv', index=False)


# In[ ]:




