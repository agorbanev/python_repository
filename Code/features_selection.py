import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, SelectFpr , SelectFwe
from sklearn.feature_selection import SelectPercentile
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import random

# main init
def get_LessFeaturesSet(X,y,method,paramlist):

    if( method =="Percentile"):
        Xnew,indexarr,scores_arr = selectionPercentile(X,y,paramlist)
        score_param =True

    if (method == "KBest"):
        Xnew,indexarr,scores_arr = selectionKBest(X,y,paramlist)

    if (method == "Fpr"):
        Xnew, indexarr,scores_arr = selectionFpr(X, y, paramlist)


    if (method == "Fwe"):
        Xnew, indexarr, scores_arr = selectionFwe(X, y, paramlist)

    if (method == "RFE"):
        Xnew, indexarr = selectionRecursiveFE(X,y,paramlist)

    if (method == "PCA"):
        Xnew, indexarr,scores_arr =  selectionPCA(X, paramlist)

    if (method == "LDA"):
        Xnew, indexarr,scores_arr =selectionLDA(X,y,paramlist)


    return [Xnew,indexarr,scores_arr]
# main finish

def selectionKBest(X,y,paramlist):
    k = paramlist['number _of_features']
    skb = SelectKBest(chi2, k=k)
    Xnew = skb.fit_transform(X, y)
    indexarr = skb.get_support(indices=True)
    scores_arr = skb.scores_
    return [Xnew,indexarr,scores_arr]

def selectionPercentile(X,y,paramlist):
    percentile =paramlist['percentile']
    spc = SelectPercentile(chi2, percentile= percentile)
    Xnew = spc.fit_transform(X, y)
    indexarr = spc.get_support(indices=True)
    scores_arr = spc.scores_
    return [Xnew,indexarr,scores_arr]


def selectionFpr(X,y,paramlist):
    k =  paramlist['number _of_features']
    fpr = SelectFpr(chi2, k=k)
    Xnew = fpr.fit_transform(X, y)
    indexarr = fpr.get_support(indices=True)
    scores_arr = fpr.scores_
    return [Xnew,indexarr,scores_arr]

def selectionFwe(X,y,paramlist):
    k =  paramlist['number _of_features']
    fwe = SelectFpr(chi2, k=k)
    Xnew = fwe.fit_transform(X, y)
    indexarr = fwe.get_support(indices=True)
    scores_arr = fwe.scores_
    return [Xnew,indexarr,scores_arr]

def selectionRecursiveFE(X,y,paramlist):
    #create estimator
    n_features_to_select = paramlist['number _of_features']
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=n_features_to_select, step=3)
    Xnew = rfe.fit_transform(X, y)
    indexarr = rfe.get_support(indices=True)
    return [Xnew, indexarr]

def selectionPCA(X,paramlist):
    n_components = paramlist['n_components']
    pca = PCA(n_components=n_components, copy=True, whiten=False,svd_solver="auto", tol = 0.0, iterated_power ="auto", random_state = None)
    #pca.fit(X)
    #Xnew = pca.transform(X)
    Xnew = pca.fit_transform(X)
    #indexarr = pca.get_support(indices=True)
    #indexarr = pca.components_
    indexarr = pca.explained_variance_ratio_
    scores_arr = None
    return [Xnew,indexarr,scores_arr]

def selectionLDA(X,y,paramlist):
    lda = LinearDiscriminantAnalysis(solver="svd", shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)
    Xnew = lda.fit_transform(X, y)
    indexarr = lda.explained_variance_ratio_
    scores_arr = None
    return [Xnew, indexarr,scores_arr]

def generate_Extra_Data(nk):

    X = np.zeros((120,7),dtype=np.float)
    farr = np.zeros(120,dtype =np.float)
    y = np.zeros(120,dtype=np.int)
    a = 0.1
    b = 0.2
    c = 0.3
    d = 0.25

    af = 0.07
    bf = 0.03
    cf =0.05

    e1 = 0.05
    e2=0.1
    e3=0.035
    e4= 0.075

    g1 = 0.7
    g2=0.1
    g3=0.05
    g4=0.15

    h1=0.12
    h2=0.33
    h3=0.3
    h4=0.25

    errcoef = 0.005
    step = 1.0/nk
    for i in range (0,120):
        f=0.0
        x1 = random.random()
        x2 = random.random()
        x3 = random.random()
        x4 = random.random()

        x5 = e1 * x1 + e2 * x2 + e3 * x3 + e4 * x4
        x6 = g1 * x1 + g2 * x2 + g3 * x3 + g4 * x4
        x7 = h1 * x1 + h2 * x2 + h3 * x3 + h4 * x4

        err = random.random()
        fv = a*x1+b*x2+c*x3+d*x4+af*x5+bf*x6+cf*x7+errcoef*(err-0.5)

        X[i, 0] = x1
        X[i, 1] = x2
        X[i, 2] = x3
        X[i, 3] = x4
        X[i, 4] = x5
        X[i, 5] = x6
        X[i, 6] = x7

        farr[i] = fv

    fmax = max(farr)
    fmin = min(farr)
    dl = fmax - fmin
    for k in range(0, 120):
        fv =(farr[k]-fmin)/dl
        for j in range(0,nk):
            if((fv>=j*step)&(fv <(j+1)*step)):
                y[k] = j

    rezlist=[X,y]

    return rezlist

if __name__ == "__main__":

    nk = 6
    X,y = generate_Extra_Data(nk)
   # method = "KBest"

    #method = "Percentile"
    #method = "RFE"

   # method = "PCA"

    method = "LDA"
    paramlist = None
    Xnew = get_LessFeaturesSet(X,y,method,paramlist)

    nk = 3
