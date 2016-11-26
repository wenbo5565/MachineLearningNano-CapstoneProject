# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 21:28:17 2016

@author: wenbma
"""

"""
This is the script to Udacity's Nano Degress Program: Machine Learning Engineer. This
problem and datset are from Kaggle competition: RedHat Business Value
"""

import pandas as pd
import numpy as np


""" Import Data """
# Please change to the appropriate path on your local machine
people_raw = pd.read_csv(r"C:\Users\wenbma\Downloads\Kaggle Sent\Kaggle Sent\people.csv")
train_raw = pd.read_csv(r"C:\Users\wenbma\Downloads\Kaggle Sent\Kaggle Sent\act_train.csv")

""" Merge people and train by people_id """
train = pd.merge(train_raw,people_raw,how='left',on=['people_id'],suffixes=('_train','_people'))

train.dtypes

"""
----------------------------------------------------------------------------------------------
                Exploratory Data Analysis
----------------------------------------------------------------------------------------------
"""

""" plot distribution of dependent variable """
train.outcome.value_counts().plot(kind='bar')

train_exclude_y = train


train_exclude_y=train_exclude_y.drop(['outcome'],axis=1)

""" plot distribution of type of independent variable """
train.dtypes.value_counts().plot(kind='bar')

train_exclude_y['people_id'].hist()

""" plot number of categories for categorical variable (excluding binary) """
category_count = {}

for column in train:
    if train[column].dtypes == object:
        category_count[column] = len(np.unique(train[column]))
        
import matplotlib.pyplot as plt
x = np.arange(len(category_count))
y = category_count.values()
plt.bar(x,y)
plt.xticks(x+0.5,category_count.keys(),rotation='vertical')
plt.show()

"""plot again but remove 'people_id','activity_id','char_10_train' and 'group_1' as they have too many categories"""
del category_count['people_id']
del category_count['activity_id']
del category_count['char_10_train']
del category_count['group_1']

x = np.arange(len(category_count))
y = category_count.values()
plt.bar(x,y)
plt.xticks(x+0.5,category_count.keys(),rotation='vertical')
plt.show()
       
"""plot again but remove "date_people" and "date_train" """
del category_count['date_people']
del category_count['date_train']

x = np.arange(len(category_count))
y = category_count.values()
plt.bar(x,y)
plt.xticks(x+0.5,category_count.keys(),rotation='vertical')
plt.show()

"""scatterplot of dependent variable (outcome) and the only numerical variable (char_38) """

plt.scatter(train['char_38'],train['outcome'],s=60,c=train['outcome'])
plt.show

"""plot distribution of outcome (0,1) colored by some categorical variable"""
 


char_6_people_0 = {}
char_6_people_1 = {}

"""compute number of observations in each category in char_6_people"""
for element,y in zip(train.char_6_people,train.outcome):    
    if y == 0 and element in char_6_people_0.keys():
        char_6_people_0[element] = char_6_people_0[element] + 1
    elif y == 0 and element not in char_6_people_0.keys():
        char_6_people_0[element] = 1
    elif y == 1 and element in char_6_people_1.keys():
        char_6_people_1[element] = char_6_people_1[element] + 1
    elif y == 1 and element not in char_6_people_1.keys():
        char_6_people_1[element] = 1

char_6_people_1['type 7'] = 0 # add missing type in char_6_people_1

n = len(char_6_people_1.keys())
ind = np.arange(n)
outcome0Group = char_6_people_0.values()
outcome1Group = char_6_people_1.values()

p1 = plt.bar(ind,outcome0Group,color='r')
p2 = plt.bar(ind,outcome1Group,color='b',bottom=outcome0Group)

plt.ylabel('Number of Observations')
plt.title('Distribution of Outcome(0 or 1) in Each Category in char_6_people')
plt.xticks(ind+0.5,char_6_people_1.keys())
plt.legend((p1[0],p2[0]),('Outcome=0','Outcome=1'),loc=2)

plt.show()
"""
-----------------------------------------------------------------------------------------------
            Data Preprocesseing
-----------------------------------------------------------------------------------------------            
"""


""" Transform boolean variable to numerical representation """
for column in train:
    if train[column].dtypes == bool:
          train[column] = train[column].astype(int) 
        
train.dtypes

""" Print number of categories for each categorical variable """
for column in train:
    if (train[column].dtypes == object) or (train[column].dtypes == bool):
        print(train[column].name)        
        print(train[column].unique().size)
    
""" 
Drop "date_train","date_people","people_id" and "activity_id" columns because there are too many categories within them.They may
cause memory error when transforming with one-hot encoding later. In addition, I cannot be sure whether they have predictive power
and date variable is always needs to be transformed to other type. So let me remove it first and might add back later after 
we quickly build our initial model
"""     

train_drop2 = train.drop(['date_train','date_people','people_id','activity_id'],axis=1)

""" One-Hot Encoder for Categorical Variable"""
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder(categorical_features=range(0,49)) # 49th variable is numerical

""" Encoder to transform categorical varibles to numerical representation"""
label = preprocessing.LabelEncoder()

"""Extract outcome from train_drop2"""
all_y = train_drop2[['outcome']]
all_x = train_drop2.drop(['outcome'],axis=1)



""" read and merge test data """
test_raw = pd.read_csv(r"C:\Users\wenbma\Downloads\Kaggle Sent\Kaggle Sent\act_test.csv")
test = pd.merge(test_raw,people_raw,how='left',on=['people_id'],suffixes=('_train','_people')) # use _train here as varaible name needs to be the same for one-hot encoding

""" convert binary variable to numerical representation """
for column in test:
    if test[column].dtypes == bool:
          test[column] = test[column].astype(int)
          
test_drop2 = test.drop(['date_train','date_people','people_id','activity_id'],axis=1)

all_x_test = test_drop2 ## change variable name for further use below

""" merge training and test set to perform one-hot encoding """

all_x_train_test = all_x.append(all_x_test)

train_all_ind = range(0,all_x.shape[0])
test_all_ind = range(all_x.shape[0],all_x.shape[0]+all_x_test.shape[0])

"""Transform categorical features to numerical representation for both training set and test set"""
for col in all_x_train_test:
    if all_x_train_test[col].dtypes==object:
        label.fit(all_x_train_test[col])
        all_x_train_test[col]=label.transform(all_x_train_test[col])

"""Transform categorical features with one-hot encoding for training set"""    
enc.fit(all_x_train_test) 
all_x_new_train_test = enc.transform(all_x_train_test)

"""Convert the resulting coo_matrix to csr_matrix"""
all_x_new_train_test_csr = all_x_new_train_test.tocsr()

"""split the transformed data back into training and test set""" 
all_x_new_test = all_x_new_train_test_csr[test_all_ind,]
all_x_new = all_x_new_train_test_csr[train_all_ind,]


""" split the data into 5 fold to be used in cross validation """
from sklearn.cross_validation import KFold
import random
random.seed(10000)
kf = KFold(len(all_y),n_folds=5)

"""
Define a function: transfrom sparse matrix to dense matrix trunk by trunk 
because transformation all at a time cause memory error
"""
def chunks(index_array, chunk_size):
    for i in xrange(0,len(index_array),chunk_size):
        yield index_array[i:i+chunk_size]


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn import tree

""" Hide Warnings """
import warnings
warnings.filterwarnings('ignore')



"""
--------------------------------------------------------------------------------------------
         cross validation for benchmark model - decision tree classifier
--------------------------------------------------------------------------------------------
"""

auc_valid = []
auc_train = []

loop_ind = 1
for train_indices, test_indices in kf:
    train_x = all_x_new.tocsr()[train_indices,]
    train_y = all_y.loc[train_indices,]
    valid_x = all_x_new.tocsr()[test_indices,]
    valid_y = all_y.loc[test_indices,]
    """Fit DecisionTree Classifier"""
    clf = tree.DecisionTreeClassifier(random_state=0,max_depth=25).fit(train_x,train_y)
    """split training and test set into chunks in order to be transformed to dense matrix"""
    list_valid_ind = list(chunks(range(0,valid_x.shape[0]),200))
    list_train_ind = list(chunks(range(0,train_x.shape[0]),200))
    
    y_hat_valid=[]
    for i in xrange(0,len(list_valid_ind)):
        y_hat_valid.extend(clf.predict(valid_x[list_valid_ind[i],:].toarray()).tolist())
    
    y_train_hat=[]
    for i in xrange(0,len(list_train_ind)):
        y_train_hat.extend(clf.predict(train_x[list_train_ind[i],:].toarray()).tolist())

    auc_valid.append(roc_auc_score(valid_y,y_hat_valid))
    auc_train.append(roc_auc_score(train_y,y_train_hat))
    print loop_ind
    loop_ind = loop_ind + 1

"""
----------------------------------------------------------------------------------
            cross validation for gradient boosting tree (initial model - untuned parameters)
-----------------------------------------------------------------------------------
"""

auc_valid = []
auc_train = []

loop_ind = 1
for train_indices, test_indices in kf:
    train_x = all_x_new.tocsr()[train_indices,]
    train_y = all_y.loc[train_indices,]
    valid_x = all_x_new.tocsr()[test_indices,]
    valid_y = all_y.loc[test_indices,]
    """Fit Gradient Boosting"""
    gbm = GradientBoostingClassifier(random_state=0).fit(train_x,train_y)
    
    """split training and test set into chunks in order to be transformed to dense matrix"""
    list_valid_ind = list(chunks(range(0,valid_x.shape[0]),200))
    list_train_ind = list(chunks(range(0,train_x.shape[0]),200))
    
    y_hat_valid=[]
    for i in xrange(0,len(list_valid_ind)):
        y_hat_valid.extend(gbm.predict(valid_x[list_valid_ind[i],:].toarray()).tolist())
    
    y_train_hat=[]
    for i in xrange(0,len(list_train_ind)):
        y_train_hat.extend(gbm.predict(train_x[list_train_ind[i],:].toarray()).tolist())

    auc_valid.append(roc_auc_score(valid_y,y_hat_valid))
    auc_train.append(roc_auc_score(train_y,y_train_hat))
    print loop_ind
    loop_ind = loop_ind + 1
    
""" 
----------------------------------------------------------------------------------------
            grid search for best parameters for [number of estimators, max_depth]
----------------------------------------------------------------------------------------

"""
n_estimators = [200] #[100,150,200]

max_depth = [50,75,100] #[5,10,15,20,25]
result = {}
loop_ind = 1
for n in n_estimators:
    for depth in max_depth:
        auc_valid = []
        auc_train = []
        for train_indices, test_indices in kf:
            train_x = all_x_new.tocsr()[train_indices,]
            train_y = all_y.loc[train_indices,]
            valid_x = all_x_new.tocsr()[test_indices,]
            valid_y = all_y.loc[test_indices,]
            """Fit Gradient Boosting"""
            gbm = GradientBoostingClassifier(random_state=0,n_estimators=n,max_depth=depth).fit(train_x,train_y)
            
            """split training and test set into chunks in order to be transformed to dense matrix"""
            list_valid_ind = list(chunks(range(0,valid_x.shape[0]),200))
            list_train_ind = list(chunks(range(0,train_x.shape[0]),200))
            
            y_hat_valid=[]
            for i in xrange(0,len(list_valid_ind)):
                y_hat_valid.extend(gbm.predict(valid_x[list_valid_ind[i],:].toarray()).tolist())
            
            y_train_hat=[]
            for i in xrange(0,len(list_train_ind)):
                y_train_hat.extend(gbm.predict(train_x[list_train_ind[i],:].toarray()).tolist())
        
            auc_valid.append(roc_auc_score(valid_y,y_hat_valid))
            auc_train.append(roc_auc_score(train_y,y_train_hat))
        key1 = (n,depth,"valid")
        key2 = (n,depth,"train")
        result[key1] = sum(auc_valid)/float(len(auc_valid)) 
        result[key2] = sum(auc_train)/float(len(auc_train))
        print loop_ind
        loop_ind = loop_ind + 1

"""
-------------------------------------------------------------------------------------------
        fitting bechmark model on the entire training set and output to predcition on the test set
-------------------------------------------------------------------------------------------
"""

clf_final = tree.DecisionTreeClassifier(random_state=0,max_depth=25).fit(all_x_new,all_y)

test_ind = list(chunks(range(0,all_x_test.shape[0]),200))

test_result_benchmark = []

""" convert from coo matrix to csr matrix """
all_x_new_test = all_x_new_test.tocsr()

""" make prediction """
for i in xrange(0,len(test_ind)):
    test_result_benchmark.extend(clf_final.predict(all_x_new_test[test_ind[i],:].toarray()).tolist())
    
result_benchmark = map(int,test_result_benchmark)

result_series = pd.Series(result_benchmark,name='outcome')
activity_id = test['activity_id']
result_df_benchmark = pd.concat([activity_id,result_series],axis=1)

""" write result to csv file """
result_df_benchmark.to_csv('output_benchmark.csv',sep=',')


"""
-----------------------------------------------------------------------------------------
        fitting gradient boosting trees with entire training set and best tuned parameters
-----------------------------------------------------------------------------------------
"""

gbm_final = GradientBoostingClassifier(random_state=0,n_estimators=200,max_depth=30).fit(all_x_new,all_y)




""" make prediction """


test_ind = list(chunks(range(0,all_x_test.shape[0]),200))

test_result = []

""" convert from coo matrix to csr matrix """
all_x_new_test = all_x_new_test.tocsr()

""" make prediction """
for i in xrange(0,len(test_ind)):
    test_result.extend(gbm_final.predict(all_x_new_test[test_ind[i],:].toarray()).tolist())
    
result = map(int,test_result)

result_series = pd.Series(result,name='outcome')
activity_id = test['activity_id']
result_df = pd.concat([activity_id,result_series],axis=1)

""" write result to csv file """
result_df.to_csv('output.csv',sep=',')

"""
--------------------------------------------------------------------------------------------
        The following is added after the second review in order to 
    (1) split the original data set into small new training set and new test set
    (2) train a model with (200,30) as parameter with the new training set and predict on the new test set
    (3) create a confustion matrix with the predcited class and true class on the new test set
--------------------------------------------------------------------------------------------
"""

""" stratified split into new training and test set """
from sklearn.cross_validation import train_test_split
X_new_train, X_new_test, Y_new_train, Y_new_test = train_test_split(all_x_new, all_y,test_size=0.2,random_state=0,stratify=all_y)

""" train the model on the new training set """
gbm_final = GradientBoostingClassifier(random_state=0,n_estimators=200,max_depth=30).fit(X_new_train,Y_new_train)




""" make prediction """


test_ind = list(chunks(range(0,X_new_test.shape[0]),200))

test_result = []

""" convert from coo matrix to csr matrix """
X_new_test = X_new_test.tocsr()

""" make prediction """
for i in xrange(0,len(test_ind)):
    test_result.extend(gbm_final.predict(X_new_test[test_ind[i],:].toarray()).tolist())

result = map(int, test_result)   
""" create confusion matrix on the new test set """
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_new_test, result)

"""
-----------------------------------------------------------------------------------------------
        Visualize Confusion Matrix by A Heatmap
-----------------------------------------------------------------------------------------------
"""

import plotly.plotly as py
py.sign_in('wenbo5565','HwoyFhKNNytDqVjYSR3q')
import plotly.graph_objs as go
from plotly.graph_objs import *
from plotly.tools import FigureFactory as FF
z = [[1752,193348],[224267,20092]]
z_text = [['False Negative:1752','True Negative:193348'],['True Positive:224267','False Positive:20092']]
x = ['True: 1', 'True: 0']
y = ['Predicted: 0','Predicted: 1']

fig = FF.create_annotated_heatmap(z,x=x,y=y,annotation_text=z_text)
plot_url = py.plot(fig)