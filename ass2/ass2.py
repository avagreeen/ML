import numpy as np
from sklearn import preprocessing as pp
#from sklearn import cross_validation as cv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

data = np.genfromtxt(open('magic04.data','rb'),delimiter=',')

data = data[:,range(10)]

print (data[[1,3],])

data = pp.scale(data)
print (data[[1,3],])


#scale data
selector = ExtraTreesClassifier(compute_importances=True, random_state=0)
data = selector.transform(data)
