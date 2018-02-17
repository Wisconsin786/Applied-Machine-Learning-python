# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 22:24:31 2018

@author: ammara
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

#print(cancer.DESCR) # Print the data set description

cancer.keys()

# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the number of features of the breast cancer dataset, which is an integer. 
    # The assignment question description will tell you the general format the autograder is expecting
    return len(cancer['feature_names'])

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
answer_zero() 

##############################################################################################
# question one 
#converting dataset with keys to dataframe 

##############################################################################################
def answer_one():
    
    # Your code here
    
    return # Return your answer


##############################################################################################
#question#1
#converting data from dic to dataframe

def answer_one():
    colum=cancer.feature_names
    t=pd.DataFrame(cancer.data) 
    t.columns=colum
    tar=cancer.target #just target column of data series 
    t.insert(30,'target',tar)
    return t
answer_one()

#################################################################################################

#question#2
#appending y variable (target=values and labels=malignant, benign)

def answer_two():
    cancerdf = answer_one()
    tar=cancer.target
    cancerdf.insert(0,'target_names',tar)
    l=[]
    m=0# correct position of count is important, it should be out of loop to define variable
    n=0
    for i in cancerdf.target:
        if i==0:
            l.append('malignant')
            m=m+1                    
        else:
            l.append('benign')
            n=n+1
    cancerdf.target_names=l # corect position is very important
    z=pd.Series([m,n],cancerdf.target_names.unique())
    return (z)    
answer_two()

##########################################################################################
#question#3
#dividing data into x an y variable

def answer_three():
    cancerdf = answer_one()
    X=cancerdf.iloc[:,0:30]  #column indexing select all columns including 1 and 31 
    y=cancerdf.target
    z=(X,y)
    return z
answer_three()

##########################################################################################

##question#4
#Train test split

from sklearn.model_selection import train_test_split
def answer_four():
    X, y = answer_three()
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0) 
    z2=(X_train, X_test, y_train, y_test)
    return z2
answer_four()
########################################################################################
##question#5
#fit a classifier

from sklearn.neighbors import KNeighborsClassifier
def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn=KNeighborsClassifier(n_neighbors=1)
    return (knn.fit(X_train, y_train))
answer_five()
###########################################################################################
#question 6
#predict label of new data (in this case mean of whole data [x features] based on Knn classifier

def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1,-1)
    knn=answer_five()  #providing it with training data
    feat_prediction=knn.predict(means)
    return (feat_prediction)  # looks like cancer is Benign
#    cancer.target_names[feat_prediction[0]] # extra line to get label
answer_six()


###########################################################################################
#question 7
#predict label of test data 
def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn=answer_five()  #providing it with training data, such as our clasifier learned from training data
    feat_prediction=knn.predict(X_test)
    return (feat_prediction) 
 #   cancer.target_names[feat_prediction] # extra line to get label
answer_seven()
##########################################################################################    
#question 8
def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    return (knn.score(X_test, y_test))
answer_eight()
###########################################################################################

#optional plot
#Try using the plotting function below to visualize the differet predicition scores between training and test sets, as well as malignant and benign cells.

#separate training and tesing data based on lables (malignant and benign tumor)

# bar graph
#score is on y axis # also for color you can just say green blue etc if you dont remeber those number with hash

#height is actually each individual score out of four total scores

import matplotlib.pyplot as plt


X_train, X_test, y_train, y_test = answer_four()

mal_train_X = X_train[y_train==0] # select only training data for malignanat tumor
mal_train_y = y_train[y_train==0] # select only training data for malignanat tumor
ben_train_X = X_train[y_train==1]
ben_train_y = y_train[y_train==1]

mal_test_X = X_test[y_test==0]
mal_test_y = y_test[y_test==0]
ben_test_X = X_test[y_test==1]
ben_test_y = y_test[y_test==1]



knn = answer_five()
scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]

# score for malignant training and benign training data; score for malignant testing and benign testing data; 
plt.figure()


# run then here then run next

bars=plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])
#plt.bar(np.arange(4), scores, color=['blue','blue','green','green'])


for bar in bars:
        height = bar.get_height() #
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)






for bar in bars:
        height = bar.get_height() #
        print(plt.gca().text(bar.get_x()))


###############################################################################
#3complete formula

def accuracy_plot():
    import matplotlib.pyplot as plt

    %matplotlib notebook

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)

accuracy_plot() 










