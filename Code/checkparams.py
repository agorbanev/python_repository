from auxil import name_exist

def checkParams(data,paramslist,isneedfile):
    length=len(paramslist)
    okstr='OK'
    valueslist=list()
    for i in range(0,length):
        keystr=paramslist[i]
        key=keystr in data
        if(not key):
            errstr='parameter '+keystr+' is absent'
            return [False,errstr,i,valueslist]
        else:
            value=data[keystr]
            valueslist.append(value)
            if(isneedfile[i]):
               filenamestr=data[keystr]
               if(not name_exist(filenamestr)):
                   errstr='File '+filenamestr + ' does not exists'
                   return [False, errstr, i,valueslist]

    return [True,okstr,length,valueslist]




