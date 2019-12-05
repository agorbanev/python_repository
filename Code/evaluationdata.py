def data_for_evalModel():


    dict1={'algorithmName':'logisticRegression','modelFile':"e:\\jsnt\\123.bin"}
    paramdict2 = dict()
    dict2 = {'algorithmName': 'kNN', 'modelFile': "e:\\jsnt\\321.bin"}
    listofdicts=list()
    listofdicts.append(dict1)
    listofdicts.append(dict2)

    paramdict={'algorithms':listofdicts,'thresholdStep':0.1,'selectionCriteria':'AUC','inputData':"e:\\jsnt\\123.csv",'progressFile':'progress.csv'}



    evaldict={'method':'modelEvaluation','parameters':paramdict,'logFile':'123.log'}

    return evaldict