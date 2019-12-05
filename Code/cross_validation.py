import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
import random
from classify_Funct import  get_Classifier


def cross_validationV1(data,labels,cross_algorithm,K,p,trainFunc,train_method,kNN,predictFunc,paramlist):

    shp = data.shape
    nr = shp[0]
    nc = shp[1]
    nclasses = len(np.unique(labels))
    probab_matrix = np.zeros((nr,nclasses),dtype=float)
    rez_score = list()

    if ( cross_algorithm =="K-fold" ):
        splitobject= KFold(n_splits=K,random_state = None, shuffle = True)

    if (cross_algorithm == "LOO"):
        split_object=LeaveOneOut()

    if (cross_algorithm == "LPO"):
        splitobject = LeavePOut(p=p)


    i=0
    for train, test in splitobject.split(data):
        X_train, X_test, y_train, y_test = data[train], data[test], labels[train], labels[test]
        clf = trainFuncV1(X_train, y_train,train_method,kNN, paramlist)
        predict_proba = predictFunc(X_test, clf)
        probab_matrix[test] = predict_proba
        ilpo_score = clf.score(X_test, y_test)
        rez_score.append(ilpo_score)
        i = i + 1


    return [probab_matrix,labels,rez_score]

def cross_validation(data,labels,cross_algorithm,K,p,trainFunc,predictFunc,paramlist):

    shp = data.shape
    nr = shp[0]
    nc = shp[1]
    nclasses = len(np.unique(labels))
    probab_matrix = np.zeros((nr,nclasses),dtype=float)
    rez_score = list()

    if ( cross_algorithm =="K-fold" ):
        #rez_score = np.zeros(K,dtype=np.float)

        kf = KFold(n_splits=K,random_state = None, shuffle = True)
        #nsp = kf.get_n_splits(data)
        #train_index, test_index = kf.split(data)
        i=0
        for train, test in kf.split(data):
            X_train, X_test, y_train, y_test = data[train], data[test], labels[train], labels[test]
            clf = trainFunc(X_train, y_train, paramlist)
            predict_proba = predictFunc(X_test, clf)
            length = len(predict_proba)
            probab_matrix[test]=predict_proba
            ikf_score = clf.score(X_test, y_test)
            rez_score.append(ikf_score)
            i = i+1

    if (cross_algorithm == "LOO"):
        loo = LeaveOneOut()
        i=0
        for train, test in loo.split(data):
            X_train, X_test, y_train, y_test = data[train], data[test], labels[train], labels[test]
            clf = trainFunc(X_train, y_train, paramlist)
            predict_proba = predictFunc(X_test, clf)
            probab_matrix[test] = predict_proba
            iloo_score = clf.score(X_test, y_test)
            rez_score.append(iloo_score)
            i = i + 1

    if (cross_algorithm == "LPO"):
        lpo = LeavePOut(p=p)
        i=0
        for train, test in lpo.split(data):
            X_train, X_test, y_train, y_test = data[train], data[test], labels[train], labels[test]
            clf = trainFunc(X_train, y_train, paramlist)
            predict_proba = predictFunc(X_test, clf)
            probab_matrix[test] = predict_proba
            ilpo_score = clf.score(X_test, y_test)
            rez_score.append(ilpo_score)
            i = i + 1

    '''
    if (cross_algorithm == "Shuffle"):
        ss = ShuffleSplit(K, test_size=test_size,random_state = 0)
        i = 0
        for train, test in ss.split(X):
            X_train, X_test, y_train, y_test = data[train], data[test], labels[train], labels[test]
            clf = trainFunc(X_train, y_train, paramlist)
            predict_proba = predictFunc(X_test, clf)
            probab_matrix[test] = predict_proba
            ilpo_score = clf.score(X_test, y_test)
            rez_score.append(ilpo_score)
            i = i + 1
    '''
    return rez_score

def  trainFunc(X,y,paramlist):

    #clf = None
    clf = svm.SVC(kernel='linear', C=1,probability=True)
    clf.fit(X,y)
    return clf


def  trainFuncV1(X,y,train_method,knn,paramlist):
     clf = get_Classifier(X,y,train_method,knn,paramlist)
     clf.fit(X, y)
     return clf



def predictFunc(X,clf):
    rez=clf.predict_proba(X)
    return rez

def generate_Data(nk):

    X = np.zeros((120,4),dtype=np.float)
    y = np.zeros(120,dtype=np.int)
    a = 0.1
    b = 0.25
    c = 0.35
    d = 0.3
    errcoef = 0.05
    step = 1.0/nk
    for i in range (0,120):
        f=0.0
        x1 = random.random()
        x2 = random.random()
        x3 = random.random()
        x4 = random.random()
        err = random.random()
        f = a*x1+b*x2+c*x3+d*x4+errcoef*(err-0.5)

        X[i, 0] = x1
        X[i, 1] = x2
        X[i, 2] = x3
        X[i, 3] = x4

        for j in range(0,nk):

            if((f>=j*step)&(f<(j+1)*step)):
                y[i] = j

    rezlist=[X,y]

    return rezlist


if __name__ == "__main__":

    nk =4
    #testlist=generate_Data(nk)

    #X=testlist[0]
    #y=testlist[1]


    csvdataname = 'e:\mamm\iris.csv'
    column_indexarr = [0, 1, 2, 3]
    class_index = 4
    #data = pd.read_csv(csvdataname, header=None)
    data = pd.read_csv( csvdataname, skiprows=1, header = None)
    shp=data.shape
    nr = shp[0]
    nc=len(column_indexarr)

    X =np.zeros((nr,nc),dtype=np.float)
    y = np.zeros(nr,dtype = np.int)
    dat = data[column_indexarr]

    X=dat.values

    yd = data[class_index]
    y=yd.values


    '''
    iris = datasets.load_iris()
    # X = iris.data[:,0:2]  # we only take the first two features.
    X = iris.data
    y = iris.target
    '''
    cross_algorithm ="K-fold"
    #cross_algorithm = "LOO"
    #cross_algorithm = "LPO"
    #cross_algorithm = "Shuffle"
    K=3
    p = 2
    paramlist = None

    probab_matrix,labels,rezscore = cross_validationV1(X, y, cross_algorithm, K,p,trainFunc, predictFunc, paramlist)
    #rezscore = cross_validation(X, y, cross_algorithm, K, p, trainFunc, predictFunc, paramlist)
    nk =1

