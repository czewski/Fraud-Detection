#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import pipeline
from sklearn import tree
from sklearn.model_selection import train_test_split
from feature_engine.encoding import OneHotEncoder
import seaborn as sns

#%%
df = pd.read_csv('../data/data.csv', sep=',')
df.head()
#%%
#df.describe()
#df.info()
print(df['isFraud'].value_counts().index)
print(df['isFraud'].value_counts())
#%% AED
#plot de porcentagem de fraudes
ax = sns.countplot(x='isFraud',data=df)
plt.title('Porcentagem de Fraudes')
total = float(len(df))
for p in ax.patches:
    porcent = 100*p.get_height()/total
    ax.annotate(f'\n{porcent}', (p.get_x()+0.4, p.get_height()), ha='center', color='black', size=10)
plt.show()

#%% Normalização de dados
#data_df['age_norm']=(data_df['age']-data_df['age'].min())/(data_df['age'].max()-data_df['age'].min())

#%%
features_num = ['amount','oldbalanceOrg','oldbalanceDest', 'newbalanceDest','newbalanceOrig','isFlaggedFraud']
features_cat = ['type','nameDest','nameOrig']

target       = 'isFraud'

df[features_cat] = df[features_cat].astype(str)

#%% encoder
onehot1 = OneHotEncoder(variables=features_cat)

#%% Separando treino e validação
x = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df['isFraud']
x_train, x_validation, y_train, y_validation = train_test_split(x, y, train_size=0.7, random_state=1234)



#%% AED
#df.isna().sum()  nenhum nulo
onehot = OneHotEncoder(variables=features_cat, drop_last=False)
#onehot.fit(df[features])

#%% Treino
clf_rl = linear_model.LogisticRegression(penalty="none")
clf_rl.fit(X, df[target])

pred_rl = clf_rl.predict(X)


#%% Pipeline

#%% 