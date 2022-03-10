#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 20:32:14 2022

@author: yuhou
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Read Data
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv')
df.head()

#Convert dates to date time objects
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()

#How many observations are in each class
df['loan_status'].value_counts()


# Data Visualization
    # Loan Status by Principal and Gender 
bins = np.linspace(df.Principal.min(), df.Principal.max(), 5)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1",col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")
g.axes[0].legend()
plt.show()
    # Loan Status by Age and Gender 
bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

    #Loan Status by Day-of-week and Gender
df['dayofweek'] = df['effective_date'].dt.dayofweek
min_val = df.dayofweek.min() 
max_val = df.dayofweek.max()
val_width = max_val - min_val
n_bins = 7
bin_width = val_width/n_bins

g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(sns.histplot,'dayofweek', bins=n_bins, binrange=(min_val, max_val))
g.axes[-1].legend()
plt.xticks(np.arange(min_val-bin_width/2, max_val+bin_width/2, bin_width),
           ["",0,1,2,3,4,5,6,""])
plt.show()

# Creating a new variable showing if effective date is on weekend
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()
# Converting gender to 0,1
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()

# Selecting features from the dataset
# And making dummy variables for Education
Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

# Defining feature set and labels
X = Feature
X[0:5]
y = df['loan_status'].values
y[0:5]

#Normalize the feature set
X= preprocessing.StandardScaler().fit(X).transform(X)

#Split the training set and the testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state = 4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

#KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

Ks = 20
accuracy_knn = np.zeros((Ks))
for i in range(1,Ks+1):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    y_hat_knn = neigh.predict(X_test)
    accuracy_knn[i-1] = metrics.accuracy_score(y_test, y_hat_knn)
#Ploting different Accuracy scores for differnt number of K
plt.figure()
plt.plot(range(1,Ks+1), accuracy_knn, 'g', linewidth=0.8)
plt.xticks(np.arange(1,Ks+1,step=1))
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('Find the Best K')

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.image as mpimg


Tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=None)
Tree.fit(X_train,y_train)

#Ploting the Tree
dot_data = export_graphviz(Tree, feature_names = Feature.columns,
                          class_names = np.unique(y_train),
                          filled = True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('tree.png')
img = mpimg.imread('tree.png')
plt.figure(figsize=(200,400))
plt.imshow(img,interpolation='nearest')


#Accuracy Score
y_hat_tree = Tree.predict(X_test)
y_hat_tree
print('Decision Tree Accuracy:', metrics.accuracy_score(y_test, 
                                                      y_hat_tree))

#Support Vector Machine
from sklearn import svm

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
yhat_svm = clf.predict(X_test)
yhat_svm
#Accuracy Score for SVM
print('SVM Accuracy:',metrics.accuracy_score(y_test, yhat_svm))
print(classification_report(y_test, yhat_svm))

#the Confustion Matrix
cmatrix = confusion_matrix(y_test, yhat_svm, labels=['COLLECTION','PAIDOFF'])
plt.figure()
sns.heatmap(cmatrix, annot=True, fmt='d',xticklabels=['Collection','Paidoff'],yticklabels=['Collection','Paidoff'],cmap='Blues')
plt.title('Confusion Matrix')

#Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver = 'liblinear').fit(X_train, y_train)
yhat_lr = LR.predict(X_test)
yhat_lr
#Accuracy for Logistic Regression
print('Logistic Regression Accuracy:',metrics.accuracy_score(y_test, yhat_lr))













