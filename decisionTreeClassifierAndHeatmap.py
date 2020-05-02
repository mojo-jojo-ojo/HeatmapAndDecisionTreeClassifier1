import pandas as pd
import numpy as np
wine_data=pd.read_csv(r"F:\3]Draft Work\ML\winequality-white.csv",
                     names=["Fixed Acidity","Volatile Acidity","Citric Acid","Residual Sugar","Chlorides", "Free Sulfur Dioxide","Total Sulfur Dioxide","Density","pH","Sulphates","Alcohol","Quality"],
                     skiprows=1,sep=r'\s*;\s*', engine='python')
wine_data.head()
import matplotlib.pyplot as plt
import seaborn as sns
corrmat=wine_data.corr()
f,ax=plt.subplots(figsize=(7,7))
sns.heatmap(corrmat,vmax=.8,square=True,annot=True,fmt='.2f')
plt.show()

x=wine_data.drop('Quality',axis=1)
y=wine_data['Quality']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#test_size=0.2 as 20% of our data will be used for training

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(max_depth=5, max_features=4, criterion='entropy')
classifier

classifier.fit(x_train, y_train)

score=classifier.score(x_test,y_test)
print(score)

classifier.n_features_

classifier.feature_importances_