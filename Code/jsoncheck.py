# coding: utf8
#import os
#import json
#import io
import numpy
from sklearn import  preprocessing
from auxil import readjsonfile

def jsoncheck(jsonnamestr):

    dataFlag=False
    needlist = ['rowIdentifier', 'selectedColumns', 'target', 'description']

    try:
        variable_assignment=readjsonfile(jsonnamestr)
    except ValueError as err:
        exceptionstr = err.args[0]
        raise ValueError(exceptionstr)
        return


    demandlist=['rowIdentifier','selectedColumns','target','description']
    l=len(demandlist)
    for i in range(0,l):
        if(demandlist[i] not in variable_assignment.keys()):
            exceptionstr = 'parameter ' + demandlist[i] + ' does not exist'
            raise ValueError(exceptionstr)
            return

    columns_list=variable_assignment['selectedColumns']
    cl=len(columns_list)
    if cl==0:
        exceptionstr = 'selectedColumns names list is empty'
        raise ValueError(exceptionstr)
        return

    ins=columns_list.count(variable_assignment['rowIdentifier'])
    if(ins>0):
        exceptionstr = 'selectedColumns names contains rowIdentifier name ' + variable_assignment['rowIdentifier']
        raise ValueError(exceptionstr)
        return
    ins = columns_list.count(variable_assignment['target'])
    if (ins > 0):

        exceptionstr = 'selectedColumns names  contains target name ' + variable_assignment['target']
        raise ValueError(exceptionstr)
        return


    descriptlist = variable_assignment['description']
    ld = len(descriptlist)

    valid_type_list=['numeric','binary','categorical']
    valid_keys_list=['name','type','uniqueCount','uniqueValues']
    lk=len(valid_keys_list)
    invalidtypelist=list()
    invalidbinlist=list()
    invalidcateglist=list()
    invtype=False
    invbin=False
    invcateg=False
    for i in range(0,ld):
        currdict = descriptlist[i]
        for j in range(0,lk):
            key=valid_keys_list[j]
            key_exists= key in currdict
            if( not key_exists):
                exceptionstr = 'Error:  parameter _ ' + key + ' _ does not exists in description element N ' + str(i+1)
                raise ValueError(exceptionstr)
                return
        column_name=currdict['name']
        column_type=currdict['type']
        uniq_count=currdict['uniqueCount']
        uniq_val_list=currdict['uniqueValues']
        nu=len(uniq_val_list)
        nt=valid_type_list.count(column_type)
        if(nt==0):
            invalidtypelist.append(column_name)
            invtype=True
            #exceptionstr = 'Error: invalid type of column'
            #raise ValueError(exceptionstr)
            #return

        if ((column_type=='binary')&((uniq_count>2)|(uniq_count<1))):
            invalidbinlist.append(column_name)
            invbin = True

            #exceptionstr = 'Error: wrong number of unique values for binary column'
            #raise ValueError(exceptionstr)
            #return

        if ((column_type == 'categorical')&(nu==0)):
            invalidcateglist.append(column_name)
            invcateg=True

            #exceptionstr = 'Error: empty list of unique values for categorical column type '
            #raise ValueError(exceptionstr)
            #return
    inistr = ' Error: '
    if(invtype|invbin|invcateg):

        if(invtype):
            #invtypestr=invalidtypelist.join()
            invtypestr =str.join(inistr,invalidtypelist)
            inistr = inistr+' Invalid column type in : ' +invtypestr
        if (invbin):
            invbinstr = str.join(inistr,invalidbinlist)
            inistr=inistr+' Invalid number of values for binary columns :' +invbinstr
        if (invcateg) :
            invcategstr=str.join(inistr,invalidcateglist)
            inistr=inistr+' No data for categorical columns : '+ invcategstr

        raise ValueError(inistr)
        return
    #descriptlist = variable_assignment['description']
    #ld = len(descriptlist)
    findlist = columns_list
    findlist.append(variable_assignment['target'])
    lf = len(findlist)
    islist = list()
    for i in range(0, lf):
        findstr = findlist[i]
        findrez = False
        for j in range(0, ld):
            currdict = descriptlist[j]
            currstr = currdict['name']
            if (findstr == currstr):
                findrez = True
        islist.append(findrez)

    ns = islist.count(False)
    if (ns > 0):
        absent_list = list()
        for l in range(0, lf):
            if (not islist[l]):
                absent_list.append(findlist[l])
        prefstr = 'description array does not contains one of selectedColumns names  or target name '
        exceptionstr = prefstr + str.join(prefstr, absent_list)
        raise ValueError(exceptionstr)
        return
    encoders=json_continue_process(variable_assignment)
    return [variable_assignment,encoders]

def json_continue_process(variable_assignment):

    encoders = {}
    encodersLabel = {}
    encodersOneHot = {}

    descriptlist=variable_assignment['description']
    ld=len(descriptlist)
    for i in range(0,ld):
        currdict=descriptlist[i]
        column_type = currdict['type']
        if((column_type=='categorical')|(column_type=='binary')):
            column_name = currdict['name']
            uniq_count = currdict['uniqueCount']
            uniq_val_list = currdict['uniqueValues']
            encodersLabel[column_name] = preprocessing.LabelEncoder()
            sample = numpy.asarray(uniq_val_list)
            encodersLabel[column_name].fit(sample)
            sample = encodersLabel[column_name].transform(sample)
            encodersOneHot[column_name] = preprocessing.OneHotEncoder(sparse=False)
            encodersOneHot[column_name].fit(sample.reshape(sample.size, 1))

    encoders["label"] = encodersLabel
    encoders["onehot"] = encodersOneHot
    return encoders





if __name__ == "__main__":
    #json_name_str = 'e:\jsnt\jsontest.txt'
    json_name_str = 'e:\jsnt\jsontest_v5.txt'
    #json_name_str = "e:\\jsnt\\123.txt"
    encoders=jsoncheck(json_name_str)
    print('NORMAL FINISH')