import numpy as np
import random
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from plot_funct import save_ROC_Plot_binary
from plot_funct import plot_confusion_matrix, plot_confusion_graphic
from plot_funct import save_ROC_Plot_mult

def save_Metrics(trainmethodstr,knn,averagestr,mse,accuracy,f_score,recall,precision,filename):

    metrics_file = open(filename, "w")

    classalgstr = "Classification algorism = "+trainmethodstr+"\n"
    if ( trainmethodstr == "kNN"):
        knnstr = " k for NN = "+str(knn)+"\n"

    avstr= "average method - "+averagestr+"\n"
    metrics_file.write(avstr)

    msestr = " MSE = "+str(round(mse,3))+"\n"
    metrics_file.write(msestr)

    accurstr = ' accuracy = '+str(round(accuracy,3))+"\n"
    metrics_file.write(accurstr)

    fstr= " F_score = "+str(round(f_score,3))+"\n"
    metrics_file.write(fstr)

    recstr = " recall = "+str(round(recall,3))+"\n"
    metrics_file.write(recstr)

    prestr = " precision = "+str(round(precision,3))+"\n"
    metrics_file.write(prestr)

    metrics_file.close()

    return

def prediction_Metrics(class_names,true_y,probab_matrix, dirnamestr,trainmethodstr,knn,averagestr):

    nk = len(np.unique(true_y))
    shp =probab_matrix.shape
    nr = shp[0]

    for i in range(0,1):
        prediction = get_MaxPrediction(probab_matrix)

        mse = mean_squared_error(true_y, prediction)
        accuracy = accuracy_score(true_y,prediction)
        f_score = f1_score(true_y,prediction, average=averagestr)
        recall = recall_score(true_y,prediction,average=averagestr)
        precision = precision_score(true_y,prediction,average=averagestr)
        #metricsname = "e:/Machine_learning/predictmetrics.txt"
        metricsname = dirnamestr+"/predictmetrics.txt"
        save_Metrics(trainmethodstr,knn,averagestr, mse, accuracy, f_score, recall, precision, metricsname)

        confus_matrix = confusion_matrix(true_y, prediction)
        #pngname = "e:/Machine_learning/confusmatr.png"
        pngname = dirnamestr+"/confusmatr.png"
        #plot_confusion_matrix(confus_matrix, 0.1, 0.1, pngname,True)
        plot_confusion_graphic(pngname, confus_matrix,False, class_names, normalize=True)
        if(nk == 2):
           score= probab_matrix[: ,0]
           fpr,tpr,thresholds = fpr, tpr, thresholds = roc_curve(true_y, score, pos_label=0)
           #auc = roc_auc_score(true_y, score)
           roc_auc = auc(fpr, tpr)
           #pictname=  "e:/Machine_learning/ROC.png"
           pictname = dirnamestr+"/ROC.png"
           save_ROC_Plot_binary(fpr,tpr,roc_auc,pictname,False)
        if(nk >2):
            fpr_list = list()
            tpr_list = list()
            auc_vect = np.zeros(nk,dtype=np.float)
            for i in range(0,nk):
                score = probab_matrix[:, i]
                fpr, tpr, thresholds = fpr, tpr, thresholds = roc_curve(true_y, score, pos_label=i)
                roc_auc = auc(fpr, tpr)
                fpr_list.append(fpr)
                tpr_list.append(tpr)
                auc_vect[i] = roc_auc
            #pictname = "e:/Machine_learning/ROC.png"
            pictname = dirnamestr + "/ROC.png"
            save_ROC_Plot_mult(fpr_list, tpr_list,auc_vect,nk, pictname, False)

    return

def get_MaxPrediction(probab_matrix):
    shp = probab_matrix.shape
    nr = shp[0]
    prediction = np.zeros(nr, dtype=int)

    for i in range(0, nr):
        prob_vect = probab_matrix[i,:]
        prediction[i] = np.argmax(prob_vect)

    return prediction

def get_Prediction(probab_matrix,nk):

    shp = probab_matrix.shape
    nr = shp[1]

    prediction = np.zeros(nr,dtype=int)

    for i in range(0,nr):
        rv=random.random()
        prob_vect=probab_matrix[i,]
        prediction[i]=get_Class_Value(rv,prob_vect,nk)



    return prediction


def get_Class_Value(rv,prob_vect,nk):

     if(nk==2):
         if(rv < prob_vect[0]):
             pval = 0
         else:
             pval=1

     if(nk>2):
         lb =0.0
         pval = 0
         for j in(0,nk-1):
            rb = lb + prob_vect[j]+0.0000000001
            if((rv>=lb)&(rv < rb)):
                     pval = j+1
            else :
                lb = rb
     return pval

def generate_Probab_Matrix(nk,coef):

    probab_matrix =np.zeros((120, nk), dtype=np.float)

    prob_vect = np.zeros(nk, dtype=np.float)

    if(nk==2):
        indexarr=[0,59,119]

    if (nk == 3):
        indexarr = [0, 39, 79, 119]

    if (nk == 4):
        indexarr = [0, 29, 59, 89 ,119]

    if (nk == 5):
        indexarr = [0, 23, 47, 71 ,95,119]

    for i in range(0,120):

        for j in range (0,nk):
            prob_vect[j] = random.random()

        for j in range(0, nk):
            if((i>=indexarr[j])&(i<=indexarr[j+1])):
                prob_vect[j] = coef*prob_vect[j]
        s=sum(prob_vect)

        prob_vect = prob_vect/s
        probab_matrix[i,:]=prob_vect

    return probab_matrix

def generate_True_Y(nk):
    true_y =np.zeros(120,dtype=np.int)
    if(nk==2):
        true_y[0:59] =0
        true_y[60:119] = 1

    if (nk == 3):
        true_y[0:39] = 0
        true_y[40:79] = 1
        true_y[80:119] = 2


    if (nk == 4):
        true_y[0:29] = 0
        true_y[30:59] = 1
        true_y[60:89] = 2
        true_y[90:119] = 3

    if (nk == 5):
        true_y[0:23] = 0
        true_y[24:47] = 1
        true_y[48:71] = 2
        true_y[72:95] = 3
        true_y[96:119] = 4


    return true_y


if __name__ == "__main__":

    nk = 5
    coef = 5.0

    probab_matrix = generate_Probab_Matrix(nk,coef)

    true_y = generate_True_Y(nk)
    #averagestr = 'binary'
    averagestr ='micro'
    #averagestr = 'micro'
    dirnamestr = "e:/Machine_learning"
    trainmethodstr ="*******"
    knn=3

    prediction_Metrics(true_y, probab_matrix,dirnamestr,trainmethodstr,knn,averagestr)