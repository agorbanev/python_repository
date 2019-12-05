import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, datasets


def get_Classifier(X,y,method,k,paramlist):

    if( method =="kNN"):
        clf =k_Nearest_Neighbors_Classifier(X,y,k,paramlist)

    if( method == "SVM"):
        clf = svm_Classifier(X,y,paramlist)

    if (method == "neural_net"):
        clf = neuro_Classifier(X,y,paramlist)

    if(method == "logistic_regression"):
        clf = logistic_Classifier(X,y,paramlist)


    return clf

def test_Classifier(X_test,y_test,clf):

    prediction = clf.predict(X_test)
    #predict_proba = clf.predict_proba(X_test)
    predict_score = clf.score(X_test, y_test)

    return [prediction,predict_score]


def k_Nearest_Neighbors_Classifier(X,y,k,paramlist):

    algorithm = paramlist['algorithm']
    metric = paramlist['metric']
    p = paramlist['p']
    weights =paramlist['weights']
    knn = KNeighborsClassifier(algorithm=algorithm, leaf_size=30, metric=metric,
                         metric_params=None, n_jobs=1, n_neighbors=k, p=p,
                         weights=weights)#,probability=True)

    knn.fit(X,y)
    return knn

def svm_Classifier(X,y,paramlist):
    #clf = svm.SVC(kernel='linear', C=1)
    C=paramlist['C']
    kernel = paramlist['kernel']
    degree = paramlist['degree']
    gamma = paramlist['gamma']
    decision_function_shape = paramlist['decision_function_shape']
    clf = svm.SVC(C=C,kernel=kernel, degree = degree, gamma =gamma, coef0 = 0.0, shrinking = True, \
                  probability = True, tol = 0.001, cache_size = 200, class_weight = None, verbose = False,\
                  max_iter = -1, decision_function_shape =decision_function_shape, random_state = None)
    clf.fit(X, y)
    return clf

def neuro_Classifier(X,y,paramlist):
    activation=paramlist['activation']
    hidden_layer_sizes = paramlist['hidden_layer_sizes']
    solver = paramlist['solver']

    clf=MLPClassifier(activation=activation, alpha=1e-05, batch_size='auto',
                  beta_1=0.9, beta_2=0.999, early_stopping=False,
                  epsilon=1e-08, hidden_layer_sizes=hidden_layer_sizes, learning_rate='constant',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                  solver=solver, tol=0.0001, validation_fraction=0.1, verbose=False,
                  warm_start=False)
    clf.fit(X, y)

    return clf

def logistic_Classifier(X,y,paramlist):
    penalty = paramlist['penalty']
    solver = paramlist['solver']
    multi_class = paramlist['multi_class']
    clf = linear_model.LogisticRegression(penalty=penalty, dual=False,tol=0.0001, C=1.0, fit_intercept=True, \
                                          intercept_scaling=1, class_weight=None, random_state=None, solver=solver,\
                                          max_iter=100, multi_class=multi_class, verbose=0, warm_start=False, n_jobs=1)
    #clf = linear_model.LogisticRegression(C=1e5,probability=True)
    clf.fit(X, y)
    return clf


if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

   # method ="kNN"
   # k=15
   # kNNparamlist = {'algorithm':'auto','metric':'minkowski','p':2,'weights':'uniform'}
   # clf =  get_Classifier(X,y,method,k,kNNparamlist)

   # method = "SVM"
   # k=1
   # SVMparamlist = { 'C':1.0,'kernel':"poly",'degree':5,'gamma':"auto",'decision_function_shape':"ovr"}
   # clf = get_Classifier(X, y, method, k, SVMparamlist)



    method = "neural_net"
    k=1
    #activation: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
    #solver: {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
    netparamlist = {'activation':"relu",'solver':"lbfgs",'hidden_layer_sizes':(5,2)}
    clf = get_Classifier(X, y, method, k, netparamlist)

    #method = "logistic_regression"
    #k=1
    #solver: {‘newton - cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’},
    #logparamlist = {'C':10.0,'solver':"lbfgs",'multi_class':"ovr",'penalty':"l2",}
    #clf = get_Classifier(X, y, method, k, logparamlist)


    h=0.05
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    pngname = "e:/Machine_learning/classify.png"
    plt.savefig(pngname, format='png', dpi=100)
    plt.show()

