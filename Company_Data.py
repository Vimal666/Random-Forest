# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:48:05 2020

@author: Vimal PM
"""

#importing the neccessary libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

#loading the datasets using pd.read_csv()
Company=pd.read_csv("D:\\DATA SCIENCE\\ASSIGNMENT\\Work done\\Decison tree\\Company_data.csv")
company=Company
company.columns
#Index(['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price',
     #  'ShelveLoc', 'Age', 'Education', 'Urban', 'US']
#converting categorical variables to numerical values
Le=preprocessing.LabelEncoder()
company["ShelveLoc"]=Le.fit_transform(Company["ShelveLoc"])
company["Urban"]= Le.fit_transform(Company["Urban"])
company["US"]=Le.fit_transform(company["US"])

#next I'm going to create a bin list and I would like to convert them into categorical format
bins=[-1,6,12,18]
company["sales_status"]=pd.cut(company["Sales"],bins,labels=["low","medium","good"])
company=company.drop("Sales",axis=1)
predictors=company.iloc[:,0:10]
target=company.iloc[:,10]
#normalizing the datasets
def norm_fun(i):
    x=(i-i.min())/(i.max()-(i.min()))
    return(x)
predictor=norm_fun(predictors.iloc[:,0:])
predictor.describe()
#model building
Rf=RandomForestClassifier(n_jobs=4,oob_score=True,n_estimators=100,criterion="entropy")
#fitting my model to my splitted dataset
Rf.fit(predictor,target)
Rf.estimators_
Rf.classes_
#array(['good', 'low', 'medium']
Rf.n_classes_
#3
Rf.n_features_
#10
Rf.n_outputs_
#1
Rf.predict(predictor)
#adding my predicted variables to my original dataset
pred=Rf.predict(predictor)
company["pred"]=Rf.predict(predictor)
pd.Series(pred).value_counts()
#medium    243
#low       130
#good       27
#importing the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(company["sales_status"],company["pred"])
    #([[ 27,   0,   0],
    # [  0, 130,   0],
    # [  0,   0, 243]]
np.mean(pred==company.sales_status)
#1.0(100% accuracy)

#next I'm going to perform bagging and boosting method to check my overfitting scenario.
#for that I'm importing some neccessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
#spliting my train test data's
predictor_train,predictor_test,target_train,target_test=train_test_split(predictor,target,test_size=0.2,random_state=10)
#I'm going to build the decison tree model and I wanted to check the decison tree score for my train_test dataset
Decision=DecisionTreeClassifier()
Decision.fit(predictor_train,target_train)
#score like accuracy for train data
Decision.score(predictor_train,target_train)
#1.0(100%)
#score for test 
Decision.score(predictor_test,target_test)
#0.6125(61%)
#here I got the scores in a over fitting scenario.So to overcome this issue I'm going tu use bagging and boosting techniques
rf=RandomForestClassifier(n_estimators=10)
rf.fit(predictor_train,target_train)
rf.score(predictor_train,target_train)
#0.9875(98%)
rf.score(predictor_test,target_test)
#0.6875(68%)
bagging=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=20)
bagging.fit(predictor_train,target_train)
bagging.score(predictor_train,target_train)
#0.9375(93%)
bagging.score(predictor_test,target_test)
#0.725(72%)
#here I'm still having some difference between my train test scores.
#I'm going to build the second bagging model to make my train test score into generalized model
bagging2=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=100)
bagging2.fit(predictor_train,target_train)
bagging2.score(predictor_train,target_train)
#0.9625(96%)
bagging2.score(predictor_test,target_test)
#0.75(75%)
#building third model
bagging3=BaggingClassifier(DecisionTreeClassifier(),max_samples=1,max_features=1.0,n_estimators=200)
bagging3.fit(predictor_train,target_train)
bagging3.score(predictor_train,target_train)
#0.60625(60%)
bagging3.score(predictor_test,target_test)
#0.6125(61%)
# when I increase my sample size and number of decision trees I got generalized model having 60% of train data and 61% of test data
#next I'm going to use the boosting technique for the same overfitting issue
Boosting=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=20,learning_rate=1)
Boosting.fit(predictor_train,target_train)
Boosting.score(predictor_train,target_train)
#1.0(100)
Boosting.score(predictor_test,target_test)
#0.675
Boosting2=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100,learning_rate=1)
Boosting2.fit(predictor_train,target_train)
Boosting2.score(predictor_train,target_train)
#1.0(100%)
Boosting2.score(predictor_test,target_test)
#0.6375
Boosting3=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=500,learning_rate=6)
Boosting3.fit(predictor_train,target_train)
Boosting3.score(predictor_train,target_train)
#1.0(#100%)
Boosting3.score(predictor_test,target_test)
#0.675%(67%)
