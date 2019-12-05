from auxil import check_algorithmname
from auxil import checkParams

def modelEvaluation(data):

    param_key = 'parameters' in data
    #algnameslist = list();
    #outnameslist = list()
    if(param_key):
        paramsdict = data['parameters']
        paramslist = ['algorithms','inputData','thresholdStep','selectionCriteria']
        isneedfilelist = [False, True, False, False]
        checkrezult = checkParams(paramsdict, paramslist, isneedfilelist)
        if ((checkrezult[1]==0)|(checkrezult[1]==1)):
            exceptionstr = checkrezult[1]
            raise ValueError(exceptionstr)
            return

        else:
            if(checkrezult[1]==2):
                defaultthreshold=0.1
            if (checkrezult[1] == 3):
                default = 0.1

            dl = paramsdict['algorithms']
            length = len(dl)
            for i in range(0, length):
                algparamsdict = dl[i]
                paramslist = ['algorithmName','modelFile']
                isneedfilelist = [False, True]
                algrezult = checkParams(algparamsdict, paramslist, isneedfilelist)
                if(not algrezult[0]):
                    exceptionstr = algrezult[1]
                    raise ValueError(exceptionstr)
                    return
                else:
                    algnamestr=algparamsdict['algorithmName']
                    valid = check_algorithmname(algnamestr)
                    if (not valid):
                        exceptionstr = 'There is no algorithm with name ' + algnamestr
                        raise ValueError(exceptionstr)
                        return
        outdict=getresponsedict()
        print('modelEvaluation')
        return outdict

    else:
        exceptionstr = 'There are no parameters for modelEvaluation method'
        raise ValueError(exceptionstr)
        return



def getresponsedict():

    responsedict=\
{"totalCount":10000,
"thresholds":[0,0.5,1.0],
"bestSelected":"logisticRegression",
"algorithms":[
{"algorithmName":"logisticRegression",
"histogram":{"good":[10,20,13],"bad":[12,1,10]},
"AUC":0.9,
"KS":25,
"Gini":100,
"importanceNames":["var1","var2","var3","var4","var5"],
"importanceValues":[0.9,0.11,0.03,0.02,0.01],
"accuracies": [10,20,10],
"KSPlot":{"KS_x":[0, 50, 100],"good":[1, 2, 3],"bad":[10,20,30],"KSLocation":0.5},
"LiftPlot":{"Lift_x":[0, 50, 100],"model":[1, 2, 3],"random":[10,20,30]},
"confusionTables":[[{"predicted":"good","groundTruth":"good","pecentage":25.00,"count":10},{"predicted":"bad","groundTruth":"good","pecentage":25.00,"count":10},{"predicted":"bad","groundTruth":"bad","pecentage":25.00,"count":10},{"predicted":"good","groundTruth":"bad","pecentage":25.00,"count":10}],[{"predicted":"good","groundTruth":"good","pecentage":25.00,"count":10},{"predicted":"bad","groundTruth":"good","pecentage":25.00,"count":10},{"predicted":"bad","groundTruth":"bad","pecentage":25.00,"count":10},{"predicted":"good","groundTruth":"bad","pecentage":25.00,"count":10}],
[{"predicted":"good","groundTruth":"good","pecentage":25.00,"count":10},{"predicted":"bad","groundTruth":"good","pecentage":25.00,"count":10},{"predicted":"bad","groundTruth":"bad","pecentage":25.00,"count":10},{"predicted":"good","groundTruth":"bad","pecentage":25.00,"count":10}]],
"percentage":[{"TP":10,"FP":10,"TN":10,"FN":10},{"TP":10,"FP":10,"TN":10,"FN":10},{"TP":10,"FP":10,"TN":10,"FN":10}]
},{"algorithmName":"kNN",
"histogram":{"good":[10,20,13],"bad":[12,1,10]},
"AUC":0.5,
"KS":25,
"Gini":100,
"accuracy":0.4,
"importanceNames":["var1","var2","var3","var4","var5"],
"importanceValues":[0.9,0.11,0.03,0.02,0.01],
"accuracies": [10,20,10],
"KSPlot":{"KS_x":[0, 50, 100],"good":[1, 2, 3],"bad":[10,20,30],"KSLocation":0.5},
"LiftPlot":{"Lift_x":[0, 50, 100],"model":[1, 2, 3],"random":[10,20,30]},
"confusionTables":[[{"predicted":"good","groundTruth":"good","pecentage":25.00,"count":10},{"predicted":"bad","groundTruth":"good","pecentage":25.00,"count":10},{"predicted":"bad","groundTruth":"bad","pecentage":25.00,"count":10},{"predicted":"good","groundTruth":"bad","pecentage":25.00,"count":10}],[{"predicted":"good","groundTruth":"good","pecentage":25.00,"count":10},{"predicted":"bad","groundTruth":"good","pecentage":25.00,"count":10},{"predicted":"bad","groundTruth":"bad","pecentage":25.00,"count":10},{"predicted":"good","groundTruth":"bad","pecentage":25.00,"count":10}],
[{"predicted":"good","groundTruth":"good","pecentage":25.00,"count":10},{"predicted":"bad","groundTruth":"good","pecentage":25.00,"count":10},{"predicted":"bad","groundTruth":"bad","pecentage":25.00,"count":10},{"predicted":"good","groundTruth":"bad","pecentage":25.00,"count":10}]],
"percentage":[{"TP":10,"FP":10,"TN":10,"FN":10},{"TP":10,"FP":10,"TN":10,"FN":10},{"TP":10,"FP":10,"TN":10,"FN":10}] }] }

    return responsedict
