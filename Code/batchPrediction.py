import datetime
from auxil import name_exist
from auxil import create_default_Name
from auxil import checkParams

def batchPrediction(data):
    exception_flag=False
    exceptionstr=' '

    logstr = create_default_Name('csv')
    outdict={'outFile':logstr}

    param_key='parameters' in data

    if(param_key):

        paramdict=data['parameters']
        paramslist=['modelFile','predictedColumnName','inputData','outFile']
        isneedfilelist=[True,False,True,True]
        checkrezult=checkParams(paramdict,paramslist,isneedfilelist)
        index=checkrezult[2]
        if(index==3):
            return outdict

        if(not checkrezult[0]):
            exceptionstr = checkrezult[1]
            raise ValueError(exceptionstr)
            return
        else:
            outdict['outFile'] = paramdict['outFile']



    else:

        exceptionstr = 'There are no parameters for batchPrediction method'
        raise ValueError(exceptionstr)
        return

    print('batchPrediction')

    return outdict