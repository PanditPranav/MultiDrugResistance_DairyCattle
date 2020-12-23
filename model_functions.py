
import sklearn
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold ,cross_val_score, train_test_split, cross_val_predict
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import learning_curve
#from pandas_ml import ConfusionMatrix
import re
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.style as style

np.random.seed(1234)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split



def run_classifier(clf, k, tuned_parameters, y, cow_parameters, herd_parameters, categorical, data):
    df = data.copy()
    
    print ("""
    
    Testing and Training datasets and preprocessing
    
    """)
    ### Testing and Training datasets and preprocessing
    #features_and_label = []
    #features_and_label.append(y)
    #print(features_and_label)
    #model_data = df[features_and_label]
    X_train, X_test, y_train, y_test = train_test_split(df[cow_parameters+ herd_parameters], df[y], 
                                                    test_size=0.30, stratify = df[y])
    print(y_train.shape)
    ## processing categorical variables for training and testing data
    from sklearn import preprocessing
    for c in categorical:
        le = preprocessing.LabelEncoder()
        le.fit(df[c])
        X_train[c] = le.transform(X_train[c]) 
        X_test[c] = le.transform(X_test[c])
        #model_data[c] = le.transform(model_data[c])
        
    
    ## Running the model
    
    a = """
    
    Simple cross validation with default parameters
    
    """
    print (a)
    print(cross_val_score(clf, X_train, y_train, cv=k))
    y_pred = cross_val_predict(clf, X_train, y_train, cv=k)
    df = pd.DataFrame({'Observed':y_train, 'Predicted':y_pred})
    confusion_matrix = ConfusionMatrix(df.Observed.values, df.Predicted.values)
    print(classification_report(y_train, y_pred))
    a = """
    
    Plot shows internal validation of the random forest model with default parameters
    
    """
    print (a)
    
    confusion_matrix.plot(backend='seaborn', normalized= False, cmap='Blues',annot= True,)#fmt='d')
    plt.show()
    a = """
    
    Exrernal validation Multiple random forest model parameters are tested with k-fold cross validation (internal validation) method
    The cross validated model is then used on a "Holdout dataset" (data which is not seen by the model in cross validation)
    to understand its external validity.
    
    
    """
    print (a)
    scores = ['precision', 'recall'] 
    scores = ['precision']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()



        clf = GridSearchCV(clf, tuned_parameters, cv=k,
                           scoring='%s_macro' % score, verbose = 0)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #    print("%0.3f (+/-%0.03f) for %r"
        #          % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()

        p = pd.DataFrame({'Observed':y_test,'Predicted': clf.predict(X_test)})
        confusion_matrix = ConfusionMatrix(p.Observed, p.Predicted)
        confusion_matrix.plot(backend='seaborn', normalized= False, cmap='Blues',annot= True,)#fmt='d')
        plt.show()
        
        print(classification_report(p.Observed, p.Predicted))
        
def run_regressor(reg, k, tuned_parameters, y, cow_parameters, herd_parameters, categorical, data):
    df = data.copy()
    
    print ("""
    
    Testing and Training datasets and preprocessing
    
    """)
    
    X_train, X_test, y_train, y_test = train_test_split(df[cow_parameters+ herd_parameters], df[y], 
                                                    test_size=0.30, stratify = df[y])
    print(y_train.shape)
    ## processing categorical variables for training and testing data
    from sklearn import preprocessing
    for c in categorical:
        le = preprocessing.LabelEncoder()
        le.fit(df[c])
        X_train[c] = le.transform(X_train[c]) 
        X_test[c] = le.transform(X_test[c])
        #model_data[c] = le.transform(model_data[c])
        
    
    ## Running the model
    
    a = """
    
    Simple cross validation with default parameters
    
    """
    print (a)
    print(cross_val_score(reg, X_train, y_train, cv=k))
    y_pred = cross_val_predict(reg, X_train, y_train, cv=k)
    df = pd.DataFrame({'Observed':y_train, 'Predicted':y_pred})
    confusion_matrix = ConfusionMatrix(df.Observed.values, df.Predicted.values)
    print(classification_report(y_train, y_pred))
    a = """
    
    Plot shows internal validation of the random forest model with default parameters
    
    """
    print (a)
    
    confusion_matrix.plot(backend='seaborn', normalized= False, cmap='Blues',annot= True,)#fmt='d')
    plt.show()
    a = """
    
    Exrernal validation Multiple random forest model parameters are tested with k-fold cross validation (internal validation) method
    The cross validated model is then used on a "Holdout dataset" (data which is not seen by the model in cross validation)
    to understand its external validity.
    
    
    """
    print (a)
    scores = ['precision', 'recall'] 
    scores = ['precision']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()



        reg = GridSearchCV(reg, tuned_parameters, cv=k,
                           scoring='%s_macro' % score, verbose = 0)
        reg.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(reg.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = reg.cv_results_['mean_test_score']
        stds = reg.cv_results_['std_test_score']
        #for mean, std, params in zip(means, stds, reg.cv_results_['params']):
        #    print("%0.3f (+/-%0.03f) for %r"
        #          % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()


        y_true, y_pred = y_test, reg.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()