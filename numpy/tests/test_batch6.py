import sys
import numpy as np
import importlib.util
import os

# Get absolute path to the numpy directory
script_dir = os.path.dirname(os.path.abspath(__file__))
numpy_dir = os.path.dirname(script_dir)

def load_module(name, filename):
    path = os.path.join(numpy_dir, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod40 = load_module("linreg", "40-Linear Regression Using Normal Equation.py")
mod63 = load_module("logreg", "63-Binary Classification with Logistic Regression.py")
mod66 = load_module("jaccard", "66-Calculate Jaccard Index for Binary Classification.py")
mod64 = load_module("svm", "64-Pegasos Kernel SVM Implementation.py")
mod65 = load_module("adaboost", "65-Implement AdaBoost Fit Method.py")

linear_regression_normal_equation = mod40.linear_regression_normal_equation
predict_logistic = mod63.predict_logistic
jaccard_index = mod66.jaccard_index
pegasos_kernel_svm = mod64.pegasos_kernel_svm
adaboost_fit = mod65.adaboost_fit

def check():
    # Linear Regression Normal Eq
    X = [[1, 1], [1, 2], [1, 3]]
    y = [1, 2, 3]
    theta = linear_regression_normal_equation(X, y)
    assert len(theta) == 2
    print("LinReg Normal Eq OK")
    
    # Logistic Regression
    X_log = np.array([[1.0, 2.0], [-1.0, -2.0]])
    weights = np.array([0.5, 0.5])
    bias = 0.0
    preds = predict_logistic(X_log, weights, bias)
    assert list(preds) == [1, 0]
    print("LogReg OK")
    
    # Jaccard
    y_true = [1, 0, 1, 1]
    y_pred = [1, 1, 0, 1]
    ji = jaccard_index(y_true, y_pred)
    assert ji == 0.5 # intersection: 2, union: 4. 2/4 = 0.5
    print("Jaccard OK")
    
    # Pegasos
    data = np.array([[1.0, -1.0], [2.0, 1.0], [-1.0, -1.0], [-2.0, 1.0]])
    labels = np.array([1, 1, -1, -1])
    alphas, b = pegasos_kernel_svm(data, labels, kernel='linear', iterations=10)
    assert isinstance(alphas, list) and isinstance(b, float)
    print("Pegasos SVM OK")
    
    # Adaboost
    X_ada = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    y_ada = np.array([1, 1, -1, -1])
    clfs = adaboost_fit(X_ada, y_ada, n_clf=2)
    assert len(clfs) == 2
    print("AdaBoost OK")

if __name__ == '__main__':
    check()
