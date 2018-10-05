import numpy as np  
import pandas as pd  
from sklearn import linear_model  
from sklearn.preprocessing import OneHotEncoder  
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from sklearn import svm

def gbdt(train_X,train_Y):
    gbdt=GradientBoostingRegressor(n_estimators=500,learning_rate=0.05)
    gbdt.fit(train_X,train_Y)
    return gbdt


def gbdt_lr(train_X,train_Y,test_X,test_Y):
    gbdt_model = gbdt(train_X,train_Y)
    tree_feature = gbdt_model.apply(train_X)
    encode = oneHot(tree_feature)
    tree_feature = encode.transform(tree_feature).toarray()

    lr = LogisticRegression()
    lr.fit(tree_feature, train_Y)

    test_X = gbdt_model.apply(test_X)
    tree_feature_test = encode.transform(test_X)
    y_pred = lr.predict_proba(tree_feature_test)[:,1]

    y_test=test_Y

    print sum(abs(y_pred-y_test))/len(y_test)
    auc = metrics.roc_auc_score(test_Y, y_pred)

    precision, recall, thresholds = precision_recall_curve(test_Y, y_pred)
    plt.plot(recall, precision)
    plt.show()
    auc = metrics.roc_auc_score(test_Y, y_pred)
    print "gbdt+lr:",auc
    error = sum(abs(y_pred-test_Y))/len(y_pred)
    print 'error ', error
    #============================
    
    fpr, tpr, roc_thresholds= metrics.roc_curve(test_Y, y_pred)
#     plt.title('Receiver Operating Characteristic')
#     plt.plot(fpr, tpr, label=" auc=" + str(auc))
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.show()
#     plt.savefig("roc.png")
    return precision, recall,thresholds,y_pred,fpr,tpr,roc_thresholds
    
    
def lr(train_X,train_Y,test_X,test_Y):
    lr = LogisticRegression()
    #encode = oneHot(train_X)
    #feature = encode.transform(train_X).toarray()
    
    lr.fit(train_X, train_Y)
    
    y_pred = lr.predict_proba(test_X)[:,1]
    auc = metrics.roc_auc_score(test_Y, y_pred)
    
    print "only lr:",auc
def oneHot(datasets):
    encode = OneHotEncoder() 
    encode.fit(datasets)
    return encode
def gbdt_train(train_X,train_Y,test_X,test_Y):
    model = gbdt(train_X,train_Y)
    y_pred = model.predict(test_X)
    auc = metrics.roc_auc_score(test_Y, y_pred)
    error = sum(abs(y_pred-test_Y))/len(y_pred)
    print 'error ', error
    
    print "only gbdt:",auc
def svm_train(train_X,train_Y,test_X,test_Y):
    model = svm.SVC(kernel='rbf')
    model.fit(train_X, train_Y)
    y_pred = model.predict(test_X)
    error = sum(abs(y_pred-test_Y))/len(y_pred)
    #print error,'sss'
    return error
def gbdt_svm(train_X,train_Y,test_X,test_Y):
    
    gbdt_model = gbdt(train_X,train_Y)
    feature = gbdt_model.apply(train_X)
    #encode = oneHot(tree_feature)
    #tree_feature = encode.transform(tree_feature).toarray()
    print feature.shape
    model = svm.SVC(kernel='rbf')
    model.fit(feature, train_Y)

    test_X = gbdt_model.apply(test_X)
    #ree_feature_test = encode.transform(test_X).toarray()
    y_pred = model.predict(test_X)
    auc = metrics.roc_auc_score(test_Y, y_pred)
    error = sum(abs(y_pred-test_Y))/len(y_pred)
    print 'accuracy', auc, 'error', error
def gbdt_svm_onehot(train_X,train_Y,test_X,test_Y):
    
    gbdt_model = gbdt(train_X,train_Y)
    tree_feature = gbdt_model.apply(train_X)
    encode = oneHot(tree_feature)
    tree_feature = encode.transform(tree_feature).toarray()

    model = svm.SVC(kernel='rbf')
    model.fit(tree_feature, train_Y)

    test_X = gbdt_model.apply(test_X)
    tree_feature_test = encode.transform(test_X).toarray()
    y_pred = model.predict(tree_feature_test)
    #auc = metrics.roc_auc_score(test_Y, y_pred)
    error = sum(abs(y_pred-test_Y))/len(y_pred)
    print auc,error
def normalize_feature(X_data):
    averages=np.average(X_data,axis=0)
    X_data-=averages

    X2=np.square(X_data)
    var=np.average(X2,axis=0)
    sd=np.sqrt(var)

    X_data/=sd
    return X_data
def svm_train(train_X,train_Y,test_X,test_Y):
    
    model = svm.NuSVC(kernel='sigmoid')
    my_clf = model.fit(train_X,train_Y)
    pre = my_clf.predict(test_X)
     
    err = sum(abs(pre-test_Y))/len(test_Y)
    return err
