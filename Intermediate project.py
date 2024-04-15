import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv(r'C:\Users\manga\Downloads\creditcard.csv')
print(dataset.shape)
print(dataset.isna().sum())
print(dataset.head())
print(pd.Series(dataset['Class']))
print(sns.countplot(dataset['Class']))
corrmat = dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corrmat , vmax=0.8 , square=True)
plt.show()
print(len(dataset[dataset['Class']==0]))#valid transaction
print(len(dataset[dataset['Class']==1])) #fradulent transactions
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values
#convert imbalanced data to balanced data
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
x_res , y_res = ros.fit_resample(X,y)
print(X.shape)
print(x_res.shape)
from collections import Counter
print(Counter(y))
print(Counter(y_res))
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x_res , y_res , test_size=0.3 , random_state=42)
print(x_train.shape)
print(y_train.shape)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 641 , random_state=0)
classifier.fit(x_train , y_train)
y_pred = classifier.predict(x_test)
n_errors = (y_pred != y_test).sum()
print(n_errors)
print(y_test.shape)
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test , y_pred)
sns.heatmap(cm , annot=True)
print(accuracy_score(y_test , y_pred))
from sklearn.metrics import precision_score
print(precision_score(y_test , y_pred))
from sklearn.metrics import recall_score
recall_score(y_test , y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test , y_pred))
