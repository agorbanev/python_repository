import datetime
from config import CONFIG
from save_load import save_obj, load_obj
from auxil import name_exist
from auxil import check_algorithmname
from auxil import checkParams
from auxil import create_default_Name
from jsoncheck import jsoncheck
from memory_limits import memory_limits
from memory_limits import read_Nstr_from_Csv
from memory_limits import refineInputData
from memory_limits import train_LogisticRegression

def trainModel(data):
    path = CONFIG['default_save_path']
    param_key='parameters' in data
    algnameslist=list();
    outnameslist=list()
    if (param_key):
        paramsdict=data['parameters']
        paramslist=['algorithms','inputData','dataScheme']
        isneedfilelist = [False, True, True]
        checkrezult=checkParams(paramsdict,paramslist,isneedfilelist)
        if(not checkrezult[0]):
            exceptionstr = checkrezult[1]
            raise ValueError(exceptionstr)
            return
        else:
            dl=paramsdict['algorithms']
            length=len(dl)

            for i in range(0,length):
                algparamsdict=dl[i]
                paramslist = ['algorithmName', 'algorithmParameters','outFile' ]
                isneedfilelist = [False, False , False]
                algrezult = checkParams(algparamsdict, paramslist, isneedfilelist)

                if(algrezult[0]|(algrezult[2]==2)):
                    algnamestr=algparamsdict['algorithmName']
                    valid=check_algorithmname(algnamestr)
                    if(not valid):
                        exceptionstr = 'There is no algorithm with name '+algnamestr
                        raise ValueError(exceptionstr)
                        return

                    if (algrezult[2] == 2):
                        outstr = create_default_Name('csv')
                    else:
                        outstr=path+algparamsdict['outFile']

                    algnameslist.append(algnamestr)
                    outnameslist.append(outstr)
                else:
                    exceptionstr = algrezult[1]
                    raise ValueError(exceptionstr)
                    return

            schemename=path+paramsdict['dataScheme']
            r_list=jsoncheck(schemename)
            jsondata=r_list[0]
            encoders=r_list[1]
            inputname=path+paramsdict['inputData']
            #outname=path+paramsdict['outFile']
            algsdict = paramsdict['algorithms']
            alglength = len(algsdict)
            for i in range(0,alglength):
                outname=outnameslist[i]
                algparamsdict = dl[i]
                algorithmname=algparamsdict['algorithmName']
                algdict= algparamsdict['algorithmParameters']
                if algorithmname=='logisticRegression':
                    length_segment=memory_limits(jsondata)[0]
                    rezult=read_Nstr_from_Csv(inputname, length_segment)
                    data=rezult[0]
                    columns_names_list=rezult[1]
                    refine=refineInputData(data, jsondata, columns_names_list, encoders)
                    X = refine[0]
                    y = refine[1]
                    classifier = train_LogisticRegression(X, y,algdict)
                    savelist=[classifier,encoders]
                    save_obj(savelist,outname)
                    print('trainModel')

                listdict = list()
                currdict={'algorithmName':algorithmname,'outFile':outname}
                listdict.append(currdict)
                outdict={'models':listdict}
           
            return outdict

    else:
        exceptionstr = 'There are no parameters for trainModel method'
        raise ValueError(exceptionstr)
        return