#import datetime
import warnings
import pandas
import csv
import numpy
import numbers
import sklearn
#from sklearn import svm
from save_load import save_obj, load_obj
from sklearn.linear_model import LogisticRegression
from auxil import readjsonfile
from auxil import create_default_Name
from config import CONFIG
from auxil import  toQuantitative
from jsoncheck import jsoncheck



def memory_limits(jsondata):
    descriptlist=jsondata['description']
    number_of_columns=0
    string_size=0.0
    for i in range (0,len(descriptlist)):
        currdict=descriptlist[i]
        column_name=currdict['name']
        column_type=currdict['type']
        uniq_count = currdict['uniqueCount']
        #uniq_val_list = currdict['uniqueValues']
        #nu = len(uniq_val_list)
        if((column_type=='numeric')|(column_type=='binary')):
            number_of_columns+=1
            string_size+=8.0
        if((column_type=='categorical')&(column_name!='target')):
            number_of_columns+=uniq_count
            string_size+=8.0*uniq_count
        if ((column_type == 'categorical') & (column_name == 'target')):
            number_of_columns += 1
            string_size += 8.0

    avail_memory=1048576.0*CONFIG['max_read_memory_limit_MB']

    number_of_strings=round(avail_memory/string_size-0.5)
    return [number_of_strings,number_of_columns]


#'----------------------------------------------------------------------------------

def read_Nstr_from_Csv(namefile,length_segment):
    header=True
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
    dat = pandas.read_csv(namefile, skiprows=skip, nrows=length_segment+1, header=rowHeader)
    nRow = dat.shape[0]
    if( nRow > length_segment):
         warnings.warn('Number of rows in file is greater then given ')
         dat.drop(nRow-1, axis=0, inplace=True)
         nr=dat.shape[0]
    columns_names_list=dat.columns.values

    return [dat,columns_names_list]

#def refineInputData(data,jsonname,columns_names_list,encoders):
def refineInputData(data, jsondata, columns_names_list, encoders):
    #jsondata = readjsonfile(jsonname)
    jsonlist=jsondata['selectedColumns']
    jsontargetname=jsondata['target']
    jsonlist.append(jsontargetname)
    lc=len(columns_names_list)
    for i in range(0,lc):
        colname=columns_names_list[i]
        nc=jsonlist.count(colname)
        if(nc==0):
            data.drop(colname, axis=1, inplace=True)



    dat1=data.reindex_axis(jsonlist , axis=1 )

    descript_list=jsondata['description']
    ld=len(descript_list)
    numericColumnsId_list=list()
    categoricalColumnsId_list=list()
    binaryColumnsId_list=list()

    for i in range(0,ld):
        columns_dict=descript_list[i]
        column_name=columns_dict['name']
        column_type=columns_dict['type']

        if column_type=='numeric' :
            numericColumnsId_list.append(column_name)

        if column_type=='categorical':
            categoricalColumnsId_list.append(column_name)

        if column_type=='binary' :
            binaryColumnsId_list.append(column_name)



    columns = dat1.columns
    numericColumnsId = numpy.where([(x in  numericColumnsId_list) for x in columns])[0]
    nNumeric = numericColumnsId.size

    categoricalColumnsId = numpy.where([(x in categoricalColumnsId_list) for x in columns])[0]
    nCategorical = categoricalColumnsId.size
    categoricalNames = columns[categoricalColumnsId]

    binaryColumnsId = numpy.where([(x in binaryColumnsId_list) for x in columns])[0]
    nBinary = binaryColumnsId.size
    binaryNames = columns[binaryColumnsId]

    for dI in range(nBinary):
        try:
            intermediate = numpy.asarray(
            encoders["label"][binaryNames[dI]].transform(dat1.values[:, binaryColumnsId[dI]]))
            intermediate = intermediate.reshape(intermediate.size,1)
            dat1[binaryNames[dI]] = intermediate
        except ValueError  as err:
            errstr = str(err)
            l = errstr.find('[')
            length = len(errstr)
            substr = errstr[l:length]
            exceptionstr = ' Binary Colunmn  _ ' + binaryNames[dI] + ' _  contains unknown value _ ' + substr
            raise ValueError(exceptionstr)
            return

    for dI in range(nCategorical):
        try:
          intermediate = numpy.asarray(encoders["label"][categoricalNames[dI]].transform(dat1.values[:, categoricalColumnsId[dI]]))
          intermediate = intermediate.reshape(intermediate.size, 1)
          dat1[categoricalNames[dI]] = intermediate
        except ValueError  as err:
            errstr=str(err)
            l=errstr.find('[')
            length=len(errstr)
            substr=errstr[l:length]
            exceptionstr=' Categorical Colunmn  _ ' + categoricalNames[dI] + ' _  contains unknown value _ '+substr
            raise ValueError(exceptionstr)
            return





    y = dat1.iloc[:, dat1.shape[1] - 1]
    dat1.drop(jsontargetname, axis=1, inplace=True)
    dat = dat1.loc[:,:].apply(toQuantitative, axis=0, decComma=False)
    X = pandas.DataFrame()

    for i in range(0,dat.shape[1]):
        nn=columns[i]
        if (columns[i] in categoricalColumnsId_list):
            intemediateAll = dat.values[:, i]
            #nanCol = pandas.notnull(intemediateAll)
            intermediate = dat.values[:, i]
            result = encoders['onehot'][columns[i]].transform(intermediate.reshape(intermediate.size, 1))
            assign = numpy.empty([dat.shape[0], result.shape[1]])
            #assign.fill(numpy.nan)
            assign[:, :] = result
            X = pandas.concat([X, pandas.DataFrame(assign)], axis=1)
        else:
            X = pandas.concat([X, pandas.DataFrame(dat.values[:, i])], axis=1)

    return [X,y]


#def train_LinearRegression(X,y,savename,param_list={}):
def train_LogisticRegression(X, y, param_list={}):

    supportListParams = {'penalty','dual','tol','C','fit_intercept','intercept_scaling','class_weight',
                         'random_state','solver','max_iter', 'multi_class','verbose','warm_start','n_jobs'}
    #default_params = {'alpha': 1.0, 'fit_intercept': True, 'normalize': False, 'max_iter': None, 'tol': 0.001,
    #                  'class_weight': None, 'solver': 'auto', 'random_state': None}
    default_params ={'penalty':'l2','dual':False,'tol':0.0001,'C':1.0,'fit_intercept':True,'intercept_scaling':1,'class_weight':None,
                         'random_state':None,'solver':'liblinear','max_iter':100, 'multi_class':'ovr','verbose':0,'warm_start':False,'n_jobs':1}

    param_list_filtered = {key: param_list[key] for key in supportListParams.intersection(param_list.keys())}
    param_list_missing = {key: default_params[key] for key in supportListParams.difference(param_list.keys())}
    param_list = dict(param_list_filtered, **param_list_missing)


   # classifier = sklearn.linear_model.RidgeClassifier(copy_X=False, **param_list)
    classifier = sklearn.linear_model.LogisticRegression(**param_list)
    '''
    classifier = sklearn.linear_model.LogisticRegression(penalty=param_list['penalty'],
                                                         dual=param_list['dual'],
                                                         tol=param_list['tol'],
                                                         C=param_list['C'],
                                                         fit_intercept=param_list['fit_intercept'],
                                                         intercept_scaling=param_list['intercept_scaling'],
                                                         class_weight=param_list['class_weight'],
                                                         random_state=param_list['random_state'],
                                                         solver=param_list['solver'],
                                                         max_iter=param_list['max_iter'],
                                                         multi_class=param_list['multiclass'],
                                                          verbose=param_list['verbose'],
                                                         warm_start=param_list['warm_start'],
                                                         n_jobs=param_list['n_jobs'])
    '''

    if (2 > len(X.shape)):
        X = X.reshape(1, -1)
    #classifier.fit(X.values, y.values)
    classifier.fit(X.values, y.values)

    '''
    classifier=sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                            intercept_scaling=1, class_weight=None, random_state=None,
                                            solver='liblinear', max_iter=100, multi_class='ovr', verbose=0,
                                            warm_start=False, n_jobs=1)
    '''

    #save_obj(classifier, savename)
    return classifier

def predict_LogisticRegression(X,loadname):
    classifier=load_obj(loadname)
    if (2 > len(X.values.shape)):
        X.values = X.values.reshape(1, -1)
    # cl = load_obj(name)
    value = classifier.predict(X)
    dm = classifier.decision_function(X)
    dmshp=dm.shape
    l=len(dmshp)
    #if dm.shape[1] == 1:
    if l ==1 :
        norm = numpy.sum(numpy.exp(dm))
        score = numpy.divide(numpy.exp(dm), norm)
    else:
        if dm.shape[1] == 1:
            norm = numpy.sum(numpy.exp(dm))
            score = numpy.divide(numpy.exp(dm), norm)
        else:
            norm = numpy.sum(numpy.exp(dm), axis=1)
            norm.shape = (dm.shape[0], 1)
            score = numpy.divide(numpy.exp(dm), norm)

    return [value, score]




if __name__ == "__main__":

    #defnamestr = create_default_Name('txt')

    #print(defnamestr)

   # with (open(defnamestr,"w")) as f:
   #     f.write("SUCCESS")

   # json_name_str = 'e:\jsnt\jsontest_v5.txt'

    #jsondata=readjsonfile(json_name_str )

   # rezlist=memory_limits(jsondata)
   # rezstr='Number of strings =  ' +str(rezlist[0]) +'    Number of columns = '+ str(rezlist[1])


    #print(rezstr)
    jsonname = 'e://jsnt//tmp//j_iris.txt'
    save_load_name='e://jsnt//tmp//123.bin'

    encoders = jsoncheck(jsonname)

    jsondata=readjsonfile(jsonname)

    ml=memory_limits(jsondata)
    ns=ml[0]
    nc=ml[1]

    namefile='e:/jsnt/tmp/iris.csv'
    rr=read_Nstr_from_Csv(namefile, 100)


    refine=refineInputData(rr[0], jsonname, rr[1],encoders)

    X=refine[0]
    y=refine[1]
    classifier=train_LogisticRegression(X, y,save_load_name, param_list={})
    rezult=predict_LogisticRegression(X,save_load_name)
    value=rezult[0]
    score=rezult[1]



    print('READ_CSV')