import pandas
import numpy
from classify_vars import classify_vars
from sklearn import linear_model
import csv
import re
import numbers
import pickle
import os
from numpy.linalg import inv
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import KFold
from outliers import train_OneClassSVM, predict_OneClassSVM
from CategInfo import Categorial_Info, analize_row , unpack_indexes,Cramer_V,get_col_index
#import outliers.py

def statFunction(namefile='',length_segment=20000,parserinfo=None,decComma=False,header=True,classification = None, \
                 labeldir = '../temp', k_fold = 3,param_list=None,uniqueNumbers=None,uniqueValues=None):

    if classification is None:
        classification = classify_vars(namefile=namefile,length_segment=length_segment,parserinfo=parserinfo,
                                       decComma=decComma,header=header)


        # Header Test defines if there is a header in the file. headerStep is equal to the number of rows
        # to skip the header
    testcon = open(namefile, 'rU')
    reader = csv.reader(testcon)
    headerTest = True
    headerStep = 0

    while headerTest:

        headerTest = False

        try:
            dataR = next(reader)
        except StopIteration:
            raise ValueError("no lines containing required number of columns were detected:",
                             " the file could be empty or value column is incorrect")

        if (len(dataR) == 0):
            headerTest = True
            headerStep = headerStep + 1
            continue

        if (numpy.all([x == '' for x in dataR])):
            headerTest = True
            headerStep = headerStep + 1
            continue

    testcon.close()

    nRow = length_segment
    n = 0

    while nRow == length_segment:

        # Definition of reading parameters
        if length_segment is None:
            skip = headerStep
        else:
            skip = length_segment * n + headerStep

        if ((header) & (n == 0)):
            rowHeader = 0
        else:
            rowHeader = None

        # Read data
        dat = pandas.read_csv(namefile, skiprows=skip, nrows=length_segment, header=rowHeader)

        # Part for Outliers Detection

        '''
        shp =dat.shape
        segmlength =shp[0]
        train_length = param_list[1]
        all_length = param_list[2]
        tM = int(float(train_length)*float(segmlength)/float(all_length))
        Indexes = numpy.arange(tM,dtype=numpy.int)
        Indexes = numpy.random.permutation(tM)
        #if(n==0) all_train_dat =  numpy.empty([se,shp[1]])
        train_dat = numpy.empty([tM,shp[1]])
        train_dat = dat.iloc[Indexes,:]
       # train_data_frame = pandas.DataFrame(train_dat,None,None,None,False)
        if(n==0):
            all_train_data=train_dat.copy(0)
        else:
             all_train_data.append(train_dat)

        '''

        # Defining the name of the columns
        if (n == 0):

            columns = dat.columns
            selectCol = numpy.where([(x in classification["var_quantitative"]) for x in columns])[0]
            nameSelect = columns[selectCol]

            rl=Categorial_Info(dat, classification,uniqueNumbers, uniqueValues)
            nav=rl[0]
            ncr=int(nav*(nav-1)/2+0.00001)
            ctypearr=rl[1]
            all_val_list=rl[2]
            cramer_matr3=numpy.zeros((12,12,ncr),dtype=int)

            # set all initial arrays values to zero
            nu = numpy.zeros([selectCol.size], dtype=numpy.uint64)
            mean = numpy.zeros([selectCol.size], dtype=numpy.float64)
            M2 = numpy.zeros([selectCol.size], dtype=numpy.float64)

            m = numpy.zeros([selectCol.size,selectCol.size],dtype = numpy.uint64)
            mean1 = numpy.zeros([selectCol.size,selectCol.size],dtype = numpy.float64)
            mean2 = numpy.zeros([selectCol.size,selectCol.size],dtype = numpy.float64)
            mean12 = numpy.zeros([selectCol.size,selectCol.size],dtype = numpy.float64)

            meanSq1 = numpy.zeros([selectCol.size, selectCol.size], dtype=numpy.float64)
            meanSq2 = numpy.zeros([selectCol.size, selectCol.size], dtype=numpy.float64)

            M1 = numpy.zeros([selectCol.size, selectCol.size], dtype=numpy.float64)
            #M2 = numpy.zeros([selectCol.size, selectCol.size], dtype=numpy.float64)

            A = numpy.zeros([selectCol.size,selectCol.size],dtype = numpy.float64)
            B = numpy.zeros([selectCol.size,selectCol.size],dtype = numpy.float64)

            R2general = numpy.zeros([selectCol.size, selectCol.size], dtype=numpy.float64)
            R3 = numpy.zeros([selectCol.size, selectCol.size], dtype=numpy.float64)
            R2line = numpy.zeros([selectCol.size, selectCol.size], dtype=numpy.float64)
            MSE =  numpy.zeros([selectCol.size, selectCol.size], dtype=numpy.float64)
            MEANy = numpy.zeros([selectCol.size, selectCol.size], dtype=numpy.float64)


        data = dat.iloc[:,selectCol].apply(toQuantitative,axis=0,decComma=decComma)


        for i1 in range(1,dat.shape[0]):
            rowdat=dat.iloc[i1,:]
            rezindlist=analize_row(rowdat, ctypearr,all_val_list,nav)
            length=len(rezindlist)
            for j1 in range(0,length):
                list1=rezindlist[j1]
                indr=list1[0][0]
                indc=list1[0][1]
                ind3=list1[0][2]
                cramer_matr3[indr,indc,ind3]+=1







        # variance
        for i in range(selectCol.size):
            nu[i],mean[i],M2[i] = oline_variance(data.values[:,i] ,nu[i],mean[i],M2[i])

        # covariance
        for i in range(selectCol.size):
            for j in range(i,selectCol.size):
                m[i, j], mean1[i, j], mean2[i, j], meanSq1[i, j], meanSq2[i, j], mean12[i, j], M1[i, j], M1[j, i] = \
                    oline_naive_covariance(data.values[:, i], data.values[:, j], m[i, j],
                                           mean1[i, j],mean2[i, j], meanSq1[i, j], meanSq2[i, j], mean12[i, j], M1[i, j], M1[j, i])

                m[j, i], mean2[j, i], mean1[j, i], meanSq2[j, i], meanSq1[j, i], mean12[j, i] = \
                    m[i, j], mean1[i, j], mean2[i, j], meanSq1[i, j], meanSq2[i, j], mean12[i, j]

        n = n + 1
        nRow = dat.shape[0]

        labelsname = os.path.join(labeldir, os.path.basename(os.path.splitext(namefile)[0])) + '_segment_%d_labels.tmp' % n

        labels = load_obj(labelsname)

        data = data.assign(k=labels)

        for i in range(k_fold):

         data_save = data.loc[data['k'] == i+1]

         k_segment_name = os.path.join(labeldir, os.path.basename(os.path.splitext(namefile)[0])) + '_segment_%d_k_%d.tmp' % (n, i+1)

         save_obj(data_save, k_segment_name)
         
    # variance
    # n, mean, M2 1xlen(selectCol)
    # variance = M2 / (n )
    variance = numpy.zeros([selectCol.size], dtype=numpy.float64)
    for i in range(selectCol.size):
        variance[i] = M2[i] / (nu[i])

    # covariance
    # m, sum1, sum2, sum12 len(selectCol)xlen(selectCol)
    # covariance = (sum12 - sum1*sum2 / m) / m
    covariance = numpy.zeros([selectCol.size,selectCol.size],dtype = numpy.float64)
    for i in range(selectCol.size):
        for j in range(i,selectCol.size):

                covariance[i,j] = mean12[i,j] - mean1[i,j] * mean2[i,j]
                covariance[j, i] = covariance[i,j]

    # y - j
    # x - i
    # y = B[i,j]*x + A[i,j]
    for i in range(selectCol.size):
        for j in range(selectCol.size):

            MEANy[i,j] = mean2[i, j]

            var = M1[i,j] / (m[i,j])

            if ((i==0)&(j==2)):
                pass

            if ((i==2)&(j==0)):
                pass

            a = covariance[i,j]

            #Search for linear regression coefficients
            B[i,j] = covariance[i,j]/var
            #A[i,j] = mean[j] - B[i,j]*mean[i]
            A[i,j] = mean2[i, j] - B[i,j]*mean1[i, j]

            if ((i==2)&(j==2)):
                pass

            #Calculating the Residual sum of squares and the R square
            #SStot = mean2[j]- mean[j]*mean[j]
            #SSres = mean2[j] - 2*B[i,j]*(sum12[i,j]/m[i,j]) - 2*A[i,j]*mean[j] + (B[i,j]*B[i,j])*mean2[i] + 2*A[i,j]*B[i,j]*mean[i] + A[i,j]*A[i,j]

            SStot = meanSq2[i, j] - mean2[i, j]*mean2[i, j]
            #SSres = sumSq2[i, j] / m[i, j] - 2 * B[i, j] * (sum12[i, j] / m[i, j]) - 2 * A[i, j] * (sum2[i, j] / m[i, j]) + \
            #     (B[i, j] * B[i, j]) * (sumSq1[i, j] / m[i, j]) + 2 * A[i, j] * B[i, j] * (sum1[i, j] / m[i, j]) + A[i, j] * A[i, j]

            SSres = (meanSq2[i, j] - 2 * B[i, j] * mean12[i, j] - 2 * A[i, j] * mean2[i, j] +
                     (B[i, j] * B[i, j]) * meanSq1[i, j] + 2 * A[i, j] * B[i, j] * mean1[i, j]) + \
                    A[ i, j] * A[i, j]

            MSE[i, j] = numpy.abs(SSres)
            R2general[i, j] = 1 - numpy.abs(SSres) / SStot
            #R3[i, j] = 1 - SSres1 / SStot1
            #Rres[i, j] = SSres1

            Val1 = mean12[i, j] - mean1[i, j] *mean2[i, j]
            Val2 = (meanSq1[i, j] - mean1[i, j]*mean1[i, j])*\
                   (meanSq2[i, j] - mean2[i, j]*mean2[i, j])

            R2line[i, j] = Val1*Val1/Val2

    Rcor = numpy.sqrt(R2line)

    Rsquare = numpy.zeros(selectCol.size)
    VIF = numpy.zeros(selectCol.size)

    for k in range(selectCol.size):

        inds = numpy.array(range(selectCol.size))
        inds = inds[inds != k]

        c = numpy.array(Rcor[inds, k])

        V1 = numpy.subtract(mean12, numpy.multiply(mean1,mean2))[inds, :][:,inds]
        V2 = numpy.multiply(numpy.subtract(meanSq1, numpy.multiply(mean1,mean1)), \
                              numpy.subtract(meanSq2, numpy.multiply(mean2,mean2)))[inds, :][:,inds]

        Rxx = numpy.divide(numpy.multiply(V1, V1),V2)

        Rsquare[k] = numpy.dot(numpy.dot(c.transpose(), inv(numpy.sqrt(Rxx))), c)
        VIF[k] = float(1) / float(1 - Rsquare[k])

    mean = {'mean':mean}
    mean = pandas.DataFrame(data=mean)
    mean.index = nameSelect

    variance = {'variance': variance}
    variance = pandas.DataFrame(data = variance)
    variance.index = nameSelect

    covariance = toDataFrame(covariance,nameSelect)
    R2general = toDataFrame(R2general,nameSelect)
    R2line = toDataFrame(R2line,nameSelect)
    Rcor = toDataFrame( Rcor,nameSelect)
    MSE = toDataFrame( MSE,nameSelect)
    A = toDataFrame(A,nameSelect)
    B = toDataFrame(B,nameSelect)

    analizeVIF(VIF,Rcor)

    analizCor(Rcor)
    analizMSE(MSE,MEANy)

    result ={"mean":mean,"variance":variance,"covariance":covariance,'R^2general':R2general,'R^2line':R2line,
             'Cor':Rcor,'MSE':MSE,'coefficient':{'a':A,'b':B}}

    #train_name = '../temp/oneclassSVM_train.tmp'
    #train_name = param_list[0]
    #save_obj(all_train_data, train_name)

    col_list=unpack_indexes(ctypearr,nav,uniqueNumbers)
    cramer_list= Cramer_V(cramer_matr3,col_list, nav)
    length=len(cramer_list)
    Vmatrix=numpy.zeros((nav,nav),float)
    Vmodmatrix=numpy.zeros((nav,nav),float)
    for i in range(0,length):
        clist=cramer_list[i]
        v=clist[0][0]
        vmod=clist[0][1]
        ncol1=clist[0][2]
        ncol2=clist[0][3]
        ind1=get_col_index(ctypearr,ncol1,nav)
        ind2 = get_col_index(ctypearr, ncol2, nav)
        #ctypearr.index(ncol2)
        Vmatrix[ind1,ind2]=v
        Vmatrix[ind2, ind1] = v
        Vmodmatrix[ind1, ind2] = vmod
        Vmodmatrix[ind2, ind1] = vmod
    columns_names_list=list()
    for i in range(0,nav):
        Vmatrix[i,i]=numpy.nan
        Vmodmatrix[i,i]=numpy.nan
        ind=ctypearr[i]
        str1=columns[ind]
        columns_names_list.append(str1)

    return [result,Vmatrix,Vmodmatrix,columns_names_list]


def load_sample_oneClassSVM(sample_name, classifier_name, classification, uniqNumber):
    # load selected sample for training

    # raw_train_dat = load_obj(sample_name)
    raw_train_dat = pandas.read_csv(sample_name, skiprows=0, nrows=100, header=0)
    # raw_train_dat =td.dropna(axis=0,how='any')

    columns = raw_train_dat.columns

    selectNumCol = numpy.where([(x in classification["var_quantitative"]) for x in columns])[0]
    nameNumSelect = columns[selectNumCol]
    numerdat = raw_train_dat.iloc[:, selectNumCol]
    numshape = numerdat.shape
    nrownum = numshape[0]
    nconum = numshape[1]
    train_column = nconum

    selectCatCol = numpy.where([(x in classification["var_categorical"]) for x in columns])[0]
    nameCatSelect = columns[selectCatCol]
    categdat = raw_train_dat.iloc[:, selectCatCol]
    catshape = categdat.shape
    nrowcat = catshape[0]
    ncolcat = catshape[1]

    #  tc_dat=pandas.get_dummies(raw_train_dat,columns=nameCatSelect)
    valarray = numpy.zeros(ncolcat, dtype=int)
    for i in range(0, ncolcat):
        key = selectCatCol[i]
        val = uniqNumber[key]
        train_column += val
        valarray[i] = val

    rezmatr = numpy.zeros((nrowcat, ncolcat), dtype=int)
    for i in range(0, ncolcat):
        catarr = numpy.array(categdat.iloc[:, i])
        le = preprocessing.LabelEncoder()
        le.fit(catarr)
        rezarr = le.transform(catarr)
        rezmatr[:, i] = rezarr

    enc = OneHotEncoder(n_values=valarray)
    enc.fit(rezmatr)
    codematr = enc.transform(rezmatr)

    codeshp = codematr.shape
    nrowcode = codeshp[0]
    ncolcode = codeshp[1]

    '''
    selectBinCol = numpy.where([(x in classification["var_binary"]) for x in columns])[0]
    nameBinSelect = columns[selectBinCol]
    bindat=raw_train_dat.iloc[:,selectBinCol]
    binshape=bindat.shape
    nrowbin=binshape[0]
    ncolbin=binshape[1]


    zarray=numpy.zeros((nrowbin,ncolbin),dtype=int)

    #binarydat=pandas.DataFrame(index=range(0,nrowbin), columns=range(0,ncolbin), dtype=int, copy=False)
    binarydat = pandas.DataFrame(zarray)
    nall = nrownum + nrowbin + nrowcat

    for j in range(0,ncolbin):
       # sample=bindat[0,j]
        binarydat[0,j]=1
        for i in range(1,nrowbin):
            if(bindat[i,j]==bindat[0,j]):
                binarydat[i,j]=1
            else:
                binarydat[i,j]=-1


    #train_OneClassSVM(train_dat,classifier_name,None,True)

    '''
    return

def toQuantitative(input,decComma=False):
    # replace strings with numbers
    result = input.map(toNumber,{"decComma":decComma})
    return result

def toNumber(x,decComma=False):
    # If x is a string-written number, converts it  to numeric
    t = type(x)
    if t is str:
        if (decComma):
            y = re.sub(",", ".", x)
        else:
            y = re.sub(",", "", x)
        try:
            y = numpy.float(y)
        except BaseException:
            y = numpy.nan
    elif issubclass(t, numbers.Number):
        y = x
    else:
        y = numpy.nan
    return y

def oline_variance(data,n = 0.,mean = 0.,M2 = 0.):
    for x in data:
        if (not pandas.isnull(x)):
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta * delta2

    # variance = M2 / (n - 1)
    return n,mean,M2

def oline_naive_covariance(data1, data2, m = 0, mean1 = 0., mean2 = 0., meanSq1 = 0., meanSq2 = 0.,  mean12 = 0.,M1 = 0.,M2 = 0.):
    #data1 - numpy.array of number
    #data2 - numpy.array of number

    notNanDat1 = numpy.where(~pandas.isnull(data1))[0]
    notNanDat2 = numpy.where(~pandas.isnull(data2))[0]

    index = numpy.intersect1d(notNanDat1,notNanDat2)

    #m +=len(index)

    dat1 = data1[index]
    dat2 = data2[index]

    for i in range(dat1.size):
        m +=1
        delta = dat1[i] - mean1
        mean1 += delta / m
        delta2 = dat1[i] - mean1
        M1 += delta * delta2

        delta = dat2[i] - mean2
        mean2 += delta / m
        delta2 = dat2[i] - mean2
        M2 += delta * delta2

        delta = dat1[i]*dat2[i] - mean12
        mean12 += delta / m

        delta = dat1[i] * dat1[i] - meanSq1
        meanSq1 += delta / m

        delta = dat2[i] * dat2[i] - meanSq2
        meanSq2 += delta / m

    # covariance = (sum12 - sum1*sum2 / n) / n
    # covariance = mean12 - mean1*mean2
    return m,mean1,mean2,meanSq1,meanSq2,mean12,M1,M2

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def toDataFrame(data,columns):
    result = pandas.DataFrame(data)
    result.columns = columns
    result.index = columns
    return result

def analizCor(input):
    #input - pandas.Dataframe
    # search the same or dependent columns in dataset and print message
    data = input.values

    indent = numpy.where(data>=0.999)
    x = indent[0]
    y = indent[1]
    for i in range(x.size):
        if (x[i]!=y[i]):
            print('The table contains the same columns: "' + str(input.columns[x[i]]) + '" and "' +
                  str(input.columns[y[i]]) + '"')

    highCorr = numpy.where((data>0.9)&(data<0.999))
    x = highCorr[0]
    y = highCorr[1]
    for i in range(x.size):
        if (x[i] != y[i]):
            print('The table contains the dependent columns: "' + str(input.columns[x[i]]) + '" and "' +
                  str(input.columns[y[i]])+ '"')

def analizMSE(MSE,MEAN,trash = 0.05):
    # MSE mean square error - sigma^2
    RMSE = numpy.sqrt(MSE.values)
    coefError = RMSE /MEAN
    #for i in range(coefError.shape[0]):
    #    coefError[i,:] = coefError[i,:]/mean
    linearity = numpy.where(coefError<trash)
    x = linearity[0]
    y = linearity[1]

    for i in range(x.size):
        if (x[i]!=y[i]):
            print('The parameter "' + str((MSE.columns[x[i]]) + '" is linearly dependent on "' +
                                          str((MSE.index[y[i]])))+ '"')
def vif_selected(data):

    data = data.fillna(0)

    Ns = range(len(data.columns))

    VIF = numpy.ones(len(data.columns))

    regr = linear_model.LinearRegression()

    for ind in Ns:
        inds = [x for x in Ns if x != ind]

        y = numpy.array(data.iloc[:, ind])

        x = numpy.array(data.iloc[:, inds])

        regr.fit(x, y)

        VIF[ind] = (float(1) / float(1 - regr.score(x, y)))

    return VIF

def analizeVIF(VIF,COR):

    indent = numpy.where(VIF > 10)

    x = indent[0]

    print('high multicollinearity detected for columns: ' + ', '.join([COR.columns[x][i] for i in range(len(COR.columns[x]))]))

    indent = numpy.where((VIF < 5) & (VIF > 1))

    x = indent[0]

    print('moderate multicollinearity detected for columns: ' + ', '.join([COR.columns[x][i] for i in range(len(COR.columns[x]))]))

def k_FoldValidation(k,n):

    for i in range(k):
        dataV = pandas.DataFrame()
        dataT = pandas.DataFrame()

        for j in range(n):
            k_segment = load_obj('../temp/data_new_segment_%d_k_%d.tmp' % (j+1,i+1))
            frames = [dataV, k_segment]
            dataV = pandas.concat(frames)

        for s in numpy.array(range(k))[numpy.array(range(k))!=i]:
         for j in range(n):
            k_segment = load_obj('../temp/data_new_segment_%d_k_%d.tmp' % (j + 1, s + 1))
            frames = [dataT, k_segment]
            dataT = pandas.concat(frames)

        outV = temp_targets(dataV.shape[0])
        outT = temp_targets(dataT.shape[0])

    print('done')

def temp_targets(N):

    i = numpy.linspace(0, N, 2 + 1)
    i = numpy.array(i, dtype='uint16')

    split_arr = []
    for ind in range(i.size - 1):
        split_arr.append(i[ind + 1] - i[ind])

    outV = []
    for i in range(2):
        outV += (numpy.ones(split_arr[i], dtype=int) * i).tolist()

    numpy.random.shuffle(outV)

    return outV


if __name__ == "__main__":
     from classificChange import printDict


     if (not os.path.isdir("../temp")):
         os.mkdir("../temp")

     # First File
     cl_var_list=classify_vars(namefile="../data/data_new.csv",k_fold=3)

     classific=cl_var_list[0]
     uniqNumber = cl_var_list[1]
     uniqValues = cl_var_list[2]


     #param_list =['../temp/oneclassSVM_train1.tmp',train_length,all_length]
     result_list = statFunction(namefile="../data/data_new.csv",length_segment=20000,classification=classific,k_fold=5,param_list=None,\
                           uniqueNumbers=uniqNumber,uniqueValues=uniqValues)
     #save_obj(result, '../temp/data_new_result.tmp')

     Vmatr=result_list[1]
     Vmodmatr=result_list[2]
     columns_names=result_list[3]

     #sample_name = param_list[0]
    #sample_name='../temp/without_data_new.tmp'
     sample_name='../data/IRIS.csv'
     classifier_name = '../temp/oneclassSVM_outlier1.tmp'

     #load_sample_oneClassSVM(sample_name, classifier_name,classific,uniqNumber)




     '''
     # Second File

     cl_var_list1 = classify_vars(namefile="../data/data2_new.csv")
     res=cl_var_list1[0]
     all_length = cl_var_list1[1]
     train_length = int(0.1 * float(all_length))
     save_obj(res, '../temp/data2_new.tmp')

     classific = load_obj('../temp/data2_new.tmp')

     param_list = ['../temp/oneclassSVM_train2.tmp', train_length, all_length]
     result = statFunction(namefile="../data/data2_new.csv", classification=classific,param_list = param_list)
     save_obj(result, '../temp/data2_new_result.tmp')

     sample_name = param_list[0]
     classifier_name = '../temp/oneclassSVM_outlier2.tmp'
     #load_sample_oneClassSVM(sample_name, classifier_name)
'''
     print('FINISH')
     # print(result)


