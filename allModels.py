from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = data.drop(['Spectral Class'], axis=1)
y = data['Spectral Class']
def find_best_model(X, y):

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.94, random_state=42)

    # Define parameter grids for each model
    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    param_grid_svm = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': [1, 0.1, 0.01],
        'svc__kernel': ['linear', 'rbf']
    }

    param_grid_rf = {
        'n_estimators': [10, 50, 100],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Initialize the models with class weights to balance inbalance data,it is one of the best method and best for this data set ,insted of smote
    dt = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'), param_grid_dt, cv=3, n_jobs=-1, verbose=2)
    svm = GridSearchCV(Pipeline([('scaler', StandardScaler()), ('svc', SVC(class_weight='balanced'))]), param_grid_svm,
                       cv=3, n_jobs=-1, verbose=2)
    rf = GridSearchCV(RandomForestClassifier(class_weight='balanced'), param_grid_rf, cv=3, n_jobs=-1, verbose=2)

    # Fit the models
    dt.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Get the best estimators
    best_dt = dt.best_estimator_
    best_svm = svm.best_estimator_
    best_rf = rf.best_estimator_

    # Evaluate the models on the test set
    dt_pred = best_dt.predict(X_test)
    svm_pred = best_svm.predict(X_test)
    rf_pred = best_rf.predict(X_test)
    # Print classification reports for each model
    print("Decision Tree Performance:\n", classification_report(y_test, dt_pred))
    print("SVM Performance:\n", classification_report(y_test, svm_pred))
    print("Random Forest Performance:\n", classification_report(y_test, rf_pred))

    # Print best parameters for each model
    print("Best Parameters for Decision Tree: ", dt.best_params_)
    print("Best Parameters for SVM: ", svm.best_params_)
    print("Best Parameters for Random Forest: ", rf.best_params_)

    # Determine the best model based on test set performance (e.g., accuracy)
    dt_score = best_dt.score(X_test, y_test)
    svm_score = best_svm.score(X_test, y_test)
    rf_score = best_rf.score(X_test, y_test)
    # Compare the scores to select the best model
    best_model = None
    if dt_score >= svm_score and dt_score >= rf_score:
        best_model = best_dt
    elif svm_score >= dt_score and svm_score >= rf_score:
        best_model = best_svm
    else:
        best_model = best_rf
    # Print the best model type
    print("Best model is:", type(best_model).__name__)
    # Return all models and best estimators
    return best_dt, best_svm, best_rf, dt, svm, rf, X_train, y_train, X_test, y_test


# for further usage let, it be
best_dt, best_svm, best_rf, dt, svm, rf, X_train, y_train, X_test, y_test = find_best_model(X, y)