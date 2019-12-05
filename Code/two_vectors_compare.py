import numpy
def two_vectors_compare(prediction,truelabels,nclass):

    eqvect = (prediction==truelabels)

    uniq_arr = numpy.unique(truelabels)
    neq = sum(eqvect)
    kmatr = numpy.zeros((5,len(uniq_arr)), dtype=numpy.int)

    for i in range(0, len(uniq_arr)):
        k_value = uniq_arr[i]
        sallk = 0
        stpk = 0
        stnk = 0
        sfpk = 0
        sfnk = 0
        for j in range(0,len(prediction)):
            if(truelabels[j] == k_value):
                sallk = sallk+1
            if((prediction[j] == k_value )&(truelabels[j] == k_value)):
                stk = stpk +1
            if ((prediction[j]!= k_value) & (truelabels[j] != k_value)):
                snk = stnk + 1
            if ((prediction[j] == k_value) & (truelabels[j]!= k_value)):
                sfpk = stpk + 1
            if ((prediction[j]!= k_value) & (truelabels[j] == k_value)):
                sfnk = sfnk + 1
        kmatr[0,i] = stpk
        kmatr[1,i] = stnk
        kmatr[2, i] = sfpk
        kmatr[3, i] = sfnk
        kmatr[4,i] = sallk
    return (kmatr)


def confus_matrix(prediction,truelabels,nclass):

    diss_matr = numpy.zeros((nclass,nclass), dtype=numpy.int)
    uniq_arr = numpy.unique(truelabels)

    for i in range(0,len(truelabels)):
        vpi = prediction[i]
        vti = truelabels[i]
        itemindex = numpy.where(uniq_arr == vpi)
        ind1 = itemindex[0]
        itemindex = numpy.where(uniq_arr == vti)
        ind2 = itemindex[0]
        diss_matr[ind1,ind2] = diss_matr[ind1,ind2]+1

    return diss_matr