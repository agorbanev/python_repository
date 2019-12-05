def data_for_trainModel():

    paramdict1={'fit_intercept':False,'solver':'lbfgs','max_iter':'10'}
    dict1={'algorithmName':'logisticRegression','algorithmParameters':paramdict1,'outFile':"e:\\jsnt\\123.bin"}
    paramdict2 = dict()
    dict2 = {'algorithmName': 'kNN', 'algorithmParameters': paramdict2, 'outFile': "e:\\jsnt\\222.bin"}
    listofdicts=list()
    listofdicts.append(dict1)
    listofdicts.append(dict2)

    paramdict={'algorithms':listofdicts,'inputData':"e:\\jsnt\\123t.csv",'dataScheme':"e:\\jsnt\\123scheme.csv",'progressFile':'progress.csv'}
    traindict={'method':'trainModel','parameters':paramdict,'logFile':'123.log'}

    return traindict