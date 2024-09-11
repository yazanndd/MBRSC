from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import joblib

# Global variables for data
X_train, X_test, y_train, y_test = None, None, None, None


# Function to prepare the data and split it
def split(data):
    global X_train, X_test, y_train, y_test

    X = data.drop(['Spectral Class'], axis=1)
    y = data['Spectral Class']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# SVM Model
def svm_model():
    global X_train, X_test, y_train, y_test

    param_grid_svm = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': [1, 0.1, 0.01],
        'svc__kernel': ['linear', 'rbf']
    }

    svm = GridSearchCV(Pipeline([('scaler', StandardScaler()), ('svc', SVC(class_weight='balanced'))]), param_grid_svm,
                       cv=3, n_jobs=-1, verbose=2)
    svm.fit(X_train, y_train)

    best_svm = svm.best_estimator_
    svm_pred = best_svm.predict(X_test)

    print("SVM Performance:\n", classification_report(y_test, svm_pred))
    print("Best Parameters for SVM: ", svm.best_params_)

    return best_svm, best_svm.score(X_test, y_test)


# Decision Tree Model
def decision_tree_model():
    global X_train, X_test, y_train, y_test

    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    dt = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'), param_grid_dt, cv=3, n_jobs=-1, verbose=2)
    dt.fit(X_train, y_train)

    best_dt = dt.best_estimator_
    dt_pred = best_dt.predict(X_test)

    print("Decision Tree Performance:\n", classification_report(y_test, dt_pred))
    print("Best Parameters for Decision Tree: ", dt.best_params_)

    return best_dt, best_dt.score(X_test, y_test)


# Random Forest Model
def random_forest_model():
    global X_train, X_test, y_train, y_test

    param_grid_rf = {
        'n_estimators': [10, 50, 100],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = GridSearchCV(RandomForestClassifier(class_weight='balanced'), param_grid_rf, cv=3, n_jobs=-1, verbose=2)
    rf.fit(X_train, y_train)

    best_rf = rf.best_estimator_
    rf_pred = best_rf.predict(X_test)

    print("Random Forest Performance:\n", classification_report(y_test, rf_pred))
    print("Best Parameters for Random Forest: ", rf.best_params_)

    return best_rf, best_rf.score(X_test, y_test)


# Main function to run all models
def run_all_models(data):
    # Prepare and split the data
    split(data)

    # Train and evaluate each model
    best_svm, svm_score = svm_model()
    best_dt, dt_score = decision_tree_model()
    best_rf, rf_score = random_forest_model()

    # Compare models and find the best one
    best_model = compare_models(svm_score, dt_score, rf_score, best_svm, best_dt, best_rf)

    return best_model


# Compare Models
def compare_models(svm_score, dt_score, rf_score, best_svm, best_dt, best_rf):
    best_model = None
    if dt_score >= svm_score and dt_score >= rf_score:
        best_model = best_dt
    elif svm_score >= dt_score and svm_score >= rf_score:
        best_model = best_svm
    else:
        best_model = best_rf

    print("Best model is:", type(best_model).__name__)
    return best_model


# Function to plot confusion matrix
def plot_confusion_matrix(model):
    global X_test, y_test
    # Predict the labels for the test set
    y_pred = model.predict(X_test)
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    # Create a confusion matrix display object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


# Plot confusion matrices for the best models
def plot_all_confusion_matrices(best_svm, best_dt, best_rf):
    plot_confusion_matrix(best_svm)
    plot_confusion_matrix(best_dt )
    plot_confusion_matrix(best_rf)

