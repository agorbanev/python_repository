import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans , MiniBatchKMeans , Birch, AgglomerativeClustering
from sklearn.mixture import GaussianMixture , BayesianGaussianMixture
from sklearn import metrics
from sklearn.metrics import pairwise_distances


# main init
def clustering_Data(X,y,nclusters,method,paramlist):

    if( method == "KMeans"):
       labels = clusteringKMeans(X,nclusters,paramlist)

    if (method == "MiniBatchKMeans"):
        labels = clusteringMiniBatchKMeans(X, nclusters, paramlist)
    if (method == "Birch"):
        labels = clusteringBirch(X, nclusters, paramlist)

    if (method == "Agglomerative"):
        labels = clusteringAgglomerative(X, nclusters, paramlist)

    if (method == "GaussMixt"):
        labels = clusteringGaussMixt(X,y,nclusters,paramlist)

    if (method == "BayGaussMixt"):
        labels = clusteringBayGaussMixt(X, y, nclusters, paramlist)

    adjrand = metrics.adjusted_rand_score(y, labels)
    adjmutscore = metrics.adjusted_mutual_info_score(y,labels)
    v_measure=metrics.v_measure_score(y, labels)
    siluetcoef = metrics.silhouette_score(X, labels, metric='euclidean')
    cal_har = metrics.calinski_harabaz_score(X, labels)
    return [adjrand,adjmutscore,v_measure,siluetcoef,cal_har]
#main frinish


def clusteringKMeans(X,nclusters,paramlist):
    kmeans = KMeans(n_clusters=nclusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001,\
                    precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
    labels = kmeans.fit_predict(X)
    return labels


def clusteringMiniBatchKMeans(X,nclusters,paramlist):
    mbkmeans =MiniBatchKMeans(n_clusters=nclusters, init='k-means++', max_iter=100, batch_size=100, verbose=0, compute_labels=True,\
                              random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
    labels = mbkmeans.fit_predict(X)

    return labels

def  clusteringBirch(X, nclusters, paramlist):
     bcl = Birch(threshold=0.5, branching_factor=50, n_clusters=nclusters, compute_labels=True, copy=True)
     labels = bcl.fit_predict(X)
     return labels

def clusteringAgglomerative(X, nclusters, paramlist):
    # don't work if memory=None, need set to existing directory name where tree structure will be written
    agglo = AgglomerativeClustering(n_clusters=nclusters, affinity='euclidean', memory="e:/Machine_learning", connectivity=None,
                                    compute_full_tree='auto', linkage='ward', pooling_func=np.mean)

    #agglo.fit(X)
    labels = agglo.fit_predict(X)
    return labels


def clusteringGaussMixt(X,y,nclusters,paramlist):

    gm = GaussianMixture(n_components=nclusters, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, \
                         init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, \
                         warm_start=False, verbose=0, verbose_interval=10)

    gm.fit(X,y)
    labels = gm.predict(X)
    return labels

def clusteringBayGaussMixt(X,y,nclusters,paramlist):

    bgm = BayesianGaussianMixture(n_components=nclusters, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1,\
                                  init_params='kmeans', weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None,\
                                  mean_precision_prior=None, mean_prior=None, degrees_of_freedom_prior=None, covariance_prior=None, \
                                  random_state=None, warm_start=False, verbose=0, verbose_interval=10)
    bgm.fit(X, y)
    labels = bgm.predict(X)
    return labels
if __name__ == "__main__":

    iris = datasets.load_iris()
    #X = iris.data[:,0:2]  # we only take the first two features.
    X = iris.data
    y = iris.target
    method = "KMeans"
    #method ="MiniBatchKMeans"
    #method = "Birch"
    #method = "Agglomerative"
    #method = "GaussMixt"
    #method = "BayGaussMixt"

    nclusters = 3
    paramlist = None
    labels = clustering_Data(X, y, nclusters, method, paramlist)

    nk =1