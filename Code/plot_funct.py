import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def plot_confusion_graphic(pngname,cm,showparam, class_names,normalize=False):

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names, normalize=normalize,
                          title='Confusion matrix',cmap=plt.cm.Blues)
    plt.savefig(pngname, format='png', dpi=100)
    if( showparam):
        plt.show()


    return

def plot_confusion_matrixOld(diss_matrix,h_hor,h_vert,pngname,showparam):

    # Initialize a Figure
    shp = diss_matrix.shape
    ndiv = shp[0]
    n=ndiv

    fig = plt.figure()

    # Add Axes to the Figure
    left = h_hor
    right = 1.0-2.0*h_hor
    bottom = h_vert
    top = 1.0-2.0*h_vert


    ax = fig.add_axes([left,bottom,right,top])

    plt.text(0.35,1.025,'True Labels',size=14)
    plt.ylabel('Prediction',size=14)
    plt.xlabel('CONFUSION MATRIX',size=14)



    # Установим локатор для главных меток
    xlocator = matplotlib.ticker.NullLocator()
    ax.xaxis.set_major_locator(xlocator)
    ylocator = matplotlib.ticker.NullLocator()
    ax.yaxis.set_major_locator(ylocator)

    ax.axhline(0.01,linewidth = 4)
    ax.axvline(0.005,linewidth = 4)

    hstep = 1.0 / ndiv
    vstep = 1.0 / ndiv
    for i in range(1,ndiv):
       ax.axhline(i*hstep,linewidth = 4)
       ax.axvline(i*vstep,linewidth = 4)

    ax.axhline(0.99,linewidth = 4)
    ax.axvline(0.995,linewidth = 4)

    maxval = numpy.max(diss_matrix)

    if( maxval <100):
        ndigits = 2
    if((maxval > 99)&(maxval < 1000)):
        ndigits = 3
    if ((maxval > 999) & (maxval < 10000)):
        ndigits = 4
    if((maxval > 9999) & (maxval < 100000)):
        ndigits = 5


    digitextsize = int(360.0/(ndiv*ndigits))

    for i in range(0, n):
        for j in range(0, n):
            numstr = str(diss_matrix[n-i-1, j] )
            ytext = (i+0.25)*hstep
            xtext = (j+0.15)*vstep
            plt.text(xtext,ytext, numstr,size = digitextsize)

    plt.savefig(pngname, format='png', dpi=100)
    if( showparam):
        plt.show()


    return

def save_ROC_Plot_binary(fpr,tpr,auc,pictname,showparam):

    auc = round(auc,2)
    aucstr = " AUC = "+str(auc)
    plt.plot(fpr, tpr, lw=2,color='b', alpha=0.8)         # label='ROC') positive %d (AUC = %0.2f)' % (i, auc))

    ytext = 0.15
    xtext = 0.75
    plt.text(xtext, ytext, aucstr, size=12,color='b')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(pictname, format='png', dpi=100)
    if(showparam):
        plt.show()
    return

def save_ROC_Plot_mult(fpr_list,tpr_list,auc_vect,nk,pictname,showparam):

    color_arr = ['b','g','m','y','c','r']

    if( nk > 6):
        return

    for i in range(0,nk):
        i1=i+1
        color=color_arr[i]
        fpr = fpr_list[i]
        tpr = tpr_list[i]
        auc = auc_vect[i]
        auc = round(auc,2)
        aucstr = " AUC"+str(i1)+" = "+str(auc)
        plt.plot(fpr, tpr, lw=2,color=color, alpha=0.8)         # label='ROC') positive %d (AUC = %0.2f)' % (i, auc))

        ytext = 0.5- i*0.075
        xtext = 0.75
        plt.text(xtext, ytext, aucstr, size=12,color=color)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(pictname, format='png', dpi=100)
    if(showparam):
        plt.show()

if __name__ == "__main__":

    n = 3
    h_hor = 0.1
    h_vert = 0.1
    pngname = "e:/Machine_learning/confusmatr.png"
    diss_matrix = numpy.zeros((n, n), dtype=numpy.int)
    for i in range(0,n):
        for j in range(0,n):
            diss_matrix[i,j] = 10*(j+1)+i

    plot_confusion_matrix(diss_matrix, h_hor, h_vert, pngname)