# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 23:09:51 2020

@author: Vimal PM
"""
#importing Neccessary Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
#loading the dataset using pd.read_csv()
Fraud=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\Work done\Random Forest\Fraud_check.csv")
Fraud.columns
#Index(['Undergrad', 'Marital.Status', 'Taxable.Income', 'City.Population',
      # 'Work.Experience', 'Urban']
fraud=Fraud
#converting my categorical data's to numerical data
Le=preprocessing.LabelEncoder()
fraud["Marital.Status"]=Le.fit_transform(Fraud["Marital.Status"])
fraud["Undergrad"]=Le.fit_transform(Fraud["Undergrad"])
fraud["Urban"]=Le.fit_transform(Fraud["Urban"])
#next I'm creating a bins list to convert my taxable income to categorical data
bins=[1,30000,100000]
fraud["status"]=pd.cut(fraud["Taxable.Income"],bins,labels=["risky","good"])
fraud=fraud.drop(["Taxable.Income"],axis=1)
#spliting my input and output variables
predictors=fraud.iloc[:,0:5]
#normalizing my input variables
def norm_fun(i):
    x=(i-i.min())/(i.max()-(i.min()))
    return(x)
predictor=norm_fun(predictors)    
predictor.describe()
target=fraud.iloc[:,5]
#buiding the model
RF=RandomForestClassifier(n_estimators=100,n_jobs=4,oob_score=True,criterion="entropy")
RF.fit(predictor,target)
RF.estimators_
RF.classes_
# array(['good', 'risky']
RF.n_features_
#5
RF.n_outputs_
#1
pred=RF.predict(predictor)
fraud["pred"]=RF.predict(predictor)
pd.Series(pred).value_counts()
#good     476
#risky    124

#importing the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(fraud["status"],fraud["status"])
#array([[476,   0],
   #   [  0, 124]]
#getting the accuracy
np.mean(pred==fraud.status)
#1.0(100%)


#next I'm going to perform bagging and boosting method to check my overfitting scenario.
##for that I'm importing some neccessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

#splitting my train test data's
predictor_train,predictor_test,target_train,target_test=train_test_split(predictor,target,test_size=0.2,random_state=10)
#First I'm going to build the decision tree model to check my overfitting
DT=DecisionTreeClassifier()
DT.fit(predictor_train,target_train)
#checking the decision tree score like accuracy
#for train data
DT.score(predictor_train,target_train)
#1.0(100%)
#for test data
DT.score(predictor_test,target_test)
#0.6583333333333333(65%)
#here my train test data's in a overfitting scenario, So to overcome this issue I'm going to perform the Bagging technique and Boosting technique.
#First I'm going to use the bagging technique
#therefor I'm building my bagging model
Bagging=BaggingClassifier(DecisionTreeClassifier(),n_estimators=20,max_samples=0.5,max_features=1.0)
#fitting my model to my train dataset
Bagging.fit(predictor_train,target_train)
#getting score for train data
Bagging.score(predictor_train,target_train)
#0.9041666666666667(93%)\
#score for test data
Bagging.score(predictor_test,target_test)
#0.7583333333333333(75%)
#I can see my train test data still having some difference
#there for I'm going to build the second bagging model
Bagging2=BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,max_samples=1,max_features=1.0)
Bagging2.fit(predictor_train,target_train)
Bagging2.score(predictor_train,target_train)
#0.7958333333333333(79%)
Bagging2.score(predictor_test,target_test)
#0.7833333333333333(78%)
#Now I got my train test data in a generalized model after I increase my number of trees and sample size.
#which means when I use my 79% of train data and based on that I got 78% of test data.

#Next I'm going to perform the Boosing technique for the same scenario.
#therefor I'm going build the Boosting model
Boosting=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=20,learning_rate=1)
#fitting the model to my train dataset
Boosting.fit(predictor_train,target_train)
#score for train data
Boosting.score(predictor_train,target_train)
#1.0
Boosting.score(predictor_test,target_test)
#0.6083333333333333(60%)
#Building the second model
Boosting2=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100,learning_rate=2)
Boosting2.fit(predictor_train,target_train)
Boosting2.score(predictor_train,target_train)
#1.0(100%)
Boosting2.score(predictor_test,target_test)
#0.60(60%)
#Building 3rd model
Boosting3=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=5000,learning_rate=6)
Boosting3.fit(predictor_train,target_train)
Boosting3.score(predictor_train,target_train)
#1.0(100%)
Boosting3.score(predictor_test,target_test)
#0.6166666666666667(61%)
