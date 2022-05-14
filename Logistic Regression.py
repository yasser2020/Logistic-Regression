# -*- coding: utf-8 -*-
"""
Created on Sat May 14 00:02:41 2022

@author: Yasser Ezzat
"""

import pandas as pd
from matplotlib import pyplot as plt

# Step 1 : Data Reading and understating
df=pd.read_csv('data/images_analyzed_productivity1.csv')
# print(df.head())
# plt.scatter(df['Age'], df['Productivity'],marker='+',color='r')

# sizes=df['Productivity'].value_counts()
# plt.pie(sizes,autopct='%1.2f%%')

# Step 2: Drop Irrelevent data

df.drop(['Images_Analyzed','User'],axis=1,inplace=True)
# print(df.head())
# Step 3: Deal with Missing Values
# df.dropna()


# Step 4: Convert Non-numeric to numeric
df['Productivity'][df['Productivity']=='Good']=1
df.Productivity[df.Productivity=='Bad']=2

# Step 5: Prepare the data(define indep/dep variables)
Y=df['Productivity'].values
Y=Y.astype('int')

X=df.drop(['Productivity'],axis=1)


# Step 6 : Split data into training and testing
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=20)

#Step 7 create a model

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train,y_train)

# Step 8 Make Predication 

predication_test=model.predict(X_test)

# Step 9 : Show accuracy of model

from sklearn import metrics

acc=metrics.accuracy_score(y_test,predication_test)
print("Model Accuracy : ",acc)

#Step 10 : print Weights 


weights=pd.Series(model.coef_[0],index=X.columns.values)
print(weights)



