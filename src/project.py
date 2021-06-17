#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import pipeline
from sklearn import tree

#%%
df = pd.read_csv('../data/data.csv', sep=',')
df.head()
#%%
#df.describe()
df.info()
#%%
features = ['type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
#features_num = []
#features_cat = []
target = 'isFraud'

#%% AED

#%% Pipeline

#%% 