import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from cross_validation import cross_validationV1, trainFunc , predictFunc
from Prediction_Metrics import prediction_Metrics
from clustering import clustering_Data
from features_selection  import  get_LessFeaturesSet

def Save_Transformed_Features_Measures(outdirname,K,method,indexlist):

    filename = outdirname + '\Transformed_features_measures.txt'
    metrics_file = open(filename, "w")
    nflstr = " Cross-validation: Number_of_Folds  = " + str(K) + "\n"
    metrics_file.write(nflstr)

    methstr = " Features Transform Method = "+ method + "\n"
    metrics_file.write(methstr)

    cmp_arr = indexlist[0]
    nstr = " Number of Components = "+str(len(cmp_arr)) + "\n"
    metrics_file.write(nstr)

    for i in range(0, K):
        comp_arr = indexlist[i]
        teststr = "Test " + str(i + 1) + "\n"
        metrics_file.write(teststr)

        mstr = " Components Explaine Variance Ratio :" + "\n"
        metrics_file.write(mstr)

        explstr = str(round(comp_arr[0], 3))
        for j in range(1,len(comp_arr)):
            explstr = explstr + '   ' + str(round(comp_arr[j], 3))
        explstr = explstr + '\n'

        metrics_file.write(explstr)

    metrics_file.close()
    return



def Save_Selected_Features_Measures(outdirname, K,method,n,measure_matrix,indexlist):

    filename = outdirname + '\selection_features_measures.txt'
    metrics_file = open(filename, "w")
    nflstr = " Cross-validation: Number_of_Folds  = " + str(K) + "\n"
    metrics_file.write(nflstr)

    shp =measure_matrix.shape
    nf = shp[1]

    nfstr = " All Features Number  = " + str(nf) + "\n"
    metrics_file.write(nfstr)

    methstr = " Features Selection Method = "+ method + "\n"
    metrics_file.write(methstr)

    if (method == "Percentile"):
        nstr = " Percents of Features to Select = " + str(n)+"%" + "\n"
    else:
        nstr = " Number of Features to Select = " + str(n)+ "\n"

    metrics_file.write(nstr)



    for i in range(0, K):
        teststr = "Test " + str(i + 1) + "\n"
        metrics_file.write(teststr)
        mstr =" Features Scores :"+ "\n"
        metrics_file.write(mstr)

        accstr = str(round(measure_matrix[i,0],3))
        for j in range(1,nf):
            accstr = accstr + '   ' + str(round(measure_matrix[i,j],3))
        accstr = accstr + '\n'

        metrics_file.write(accstr)

        indstr ="Selected Features Indexes "
        metrics_file.write(indstr)
        index_arr = indexlist[i]
        istr = str(round(index_arr[0],3))
        for j in range(1,len(index_arr)):
            istr = istr + '   ' + str(round(index_arr[j],3))
        istr = istr + '\n'
        metrics_file.write(istr)

    metrics_file.close()
    return

def Save_Clustering_Measures(outdirname,K,clustering_method,number_of_clusters,measure_matrix):

    filename = outdirname+'\clustering_measures.txt'

    metrics_file = open(filename, "w")

    nflstr = " Cross-validation: Number_of_Folds "+str(K)+"\n"
    metrics_file.write(nflstr)
    algostr = "Clustering algorithm - "+clustering_method+"\n"
    metrics_file.write(algostr)
    nclstr = "Number of Clusters " + str(number_of_clusters) + "\n"
    metrics_file.write(nclstr)
    K1 =K+1
    for i in range(0,K1):
        if(i<K):
            teststr = "Test "+str(i+1)+ "\n"
        else:
            teststr = "Full Dataset Test " + "\n"
        metrics_file.write(teststr)
        adjrandstr =" Adj_Rand_Index = "+str(round(measure_matrix[i,0],3))
        multstr = " Adj_Mutual_info = "+str(round(measure_matrix[i,1],3))
        vstr = " V_Measure = "+str(round(measure_matrix[i,2],3))
        silstr = " Siluette_Coef = "+str(round(measure_matrix[i,3],3))
        chstr = " Calinski_Harabaz_Index = "+str(round(measure_matrix[i,4],3))
        rezstr=adjrandstr+"    "+multstr+"    "+vstr+"    "+silstr+"    "+chstr+"\n"
        metrics_file.write(rezstr)

    metrics_file.close()
    return

#main start
def Class_Clust_Reduct(outdirname,csvdataname,class_names,column_indexarr,class_index,taskstr,cross_valid_params,dim_reduct_params,paramlist):

    data = pd.read_csv(csvdataname, skiprows=1, header=None)
    shp = data.shape
    nr = shp[0]
    nc = len(column_indexarr)



    X = np.zeros((nr, nc), dtype=np.float)
    dat = data[column_indexarr]
    X = dat.values

    y = np.zeros(nr, dtype=np.int)
    yd = data[class_index]
    y = yd.values

    if(( taskstr == 'Classification')|( taskstr == 'classification')):
        probab_matrix, labels, rez_score = classify_With_Cross_Validation(outdirname,X,y,class_names,cross_valid_params,paramlist)

    if (( taskstr == 'Clustering')|(taskstr =='clustering')):
        clustering_With_Cross_Validation(outdirname, X, y, cross_valid_params, paramlist)

    if ((taskstr == 'feature_selection') | (taskstr == 'Feature_Selection')):
        dimension_reduction_With_CrossValidation(outdirname,X,y,class_names,cross_valid_params,dim_reduct_params,paramlist)


    return [X,y]
# main finish



def classify_With_Cross_Validation(outdirname,X,y,class_names,cross_valid_params,paramlist):
    cross_algorithm = cross_valid_params['cross_algorithm']
    K = cross_valid_params['N_folds']
    p = cross_valid_params['p']
    train_method = cross_valid_params['train_method']
    knn = cross_valid_params['kNN']
    averagestr = cross_valid_params['average']
    probab_matrix, labels, rezscore = cross_validationV1(X, y, cross_algorithm, K, p, trainFunc,train_method,knn, predictFunc, paramlist)

    prediction_Metrics(class_names,labels, probab_matrix,outdirname, train_method, knn, averagestr)
    return [probab_matrix,labels,rezscore]
    #return


def clustering_With_Cross_Validation(outdirname,X,y,cross_valid_params,paramlist):

    K = cross_valid_params['N_folds']

    K1 = K+1
    clustering_method = cross_valid_params['clustering_algorithm']
    number_of_clusters =  cross_valid_params['number_of_clusters']

    kf = KFold(n_splits=K, random_state=None, shuffle=True)

    measure_matrix = np.zeros((K1,5),dtype=np.float)

    i = 0
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        measure_matrix[i,:] = clustering_Data(X_train, y_train, number_of_clusters, clustering_method, paramlist)
        i = i + 1
    measure_matrix[i, :] = clustering_Data(X, y, K, clustering_method, paramlist)
    Save_Clustering_Measures(outdirname,K,clustering_method,number_of_clusters,measure_matrix)
    return


def dimension_reduction_With_CrossValidation(outdirname,X,y,cross_valid_params,dim_reduct_params,paramlist):
        K = cross_valid_params['N_folds']

        reduction_type = dim_reduct_params['reduction_type']
        shp = X.shape
        nf = shp[1]

        method = dim_reduct_params['selection_transformation_method']

        kf = KFold(n_splits=K, random_state=None, shuffle=True)

        measure_matrix = np.zeros((K, nf), dtype=np.float)
        index_list = list()

        i = 0
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            Xnew,index_arr,scores_arr =  get_LessFeaturesSet(X_train,y_train,method,paramlist)
            measure_matrix[i, :] = scores_arr
            index_list.append(index_arr)
            i = i + 1


        if(reduction_type == 'selection'):
            if( method =="Percentile"):
                n = paramlist['percentile']
            else:
                 n=  paramlist['number _of_features']

            Save_Selected_Features_Measures(outdirname, K,method,n,measure_matrix,index_list)

        if (reduction_type == 'transformation'):
            Save_Transformed_Features_Measures(outdirname,K,method,index_list)

        return



if __name__ == "__main__":

    csvdataname = 'e:\mamm\iris.csv'
    column_indexarr=[0,1,2,3]
    class_index = 4
    class_names =["setosa","versicolor","virginica"]
    #cross_valid_params ={'cross_algorithm':"K-fold",'N_folds':3,'p':2,'train_method':"kNN",'kNN':7}
    cross_valid_params = {'cross_algorithm': "K-fold", 'N_folds':3, 'p': 2, 'train_method': "SVM", 'kNN': 7,'average':"macro",'clustering_algorithm':"Agglomerative","number_of_clusters":6}
    taskstr = 'Classification'
    #taskstr = 'Clustering'
    #taskstr ='feature_selection'
    #cross_valid_str = "K-fold"
    paramlist = None

    outdirname = "e:/Machine_learning"
    #kNNparamlist = {'algorithm': 'auto', 'metric': 'minkowski', 'p': 2, 'weights': 'uniform'}
    SVMparamlist = {'C': 1.0, 'kernel': "poly", 'degree': 5, 'gamma': "auto", 'decision_function_shape': "ovr"}
    #dim_reduct_params ={'reduction_type':"selection" ,'selection_method':"Percentile"}

    dim_reduct_params = {'reduction_type': "transformation", 'selection_transformation_method': "LDA"}

    selparamlist = {'number _of_features':3,'percentile':50,'n_components':3}

    X,y =  Class_Clust_Reduct(outdirname,csvdataname,class_names,column_indexarr, class_index, taskstr, cross_valid_params,dim_reduct_params,SVMparamlist)
    nk =1