import numpy as np
import math as m
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from copy import copy
from sklearn import svm

class SSLDA_Classifier():
    def __init__(self,max_iter=10,n_components=10):
        self.n_components, self.max_iter = n_components, max_iter
        self.covariance_, self.means_, self.classifier = None, None, None
        self.propagated_labels = None
        
    def predict(self, X):
        return self.classifier.predict(X)
    
    def score(self, X, y):
        return self.classifier.score(X,y)
    
    def predict_proba(self, X):
        return self.classifier.predict_proba(X)
    
    def predict_log_proba(self, X):
        return self.classifier.predict_log_proba(X)
    
    def fit(self, X,y, method='self-training', treshold=0.6):
        getLabel = lambda p: np.where(p>treshold)[0][0] if np.any(p>treshold) else -1 
        yp = copy(y)
        mask = np.ones(len(y),dtype=bool) #mask of labeled data
        mask[np.where(yp==-1)[0]] = False #cheke unlabeled data , msk = number of labeled data
        
        lda = LinearDiscriminantAnalysis(solver='svd',store_covariance=True, n_components=10)
        #print(y)
        #if there are no unlabeled data
        if(len(np.where(yp==-1)[0])==0):  #replace with len(mask)=0?
            method = 'supervised'
            
        if method =='supervised':
            lda.fit(X[mask,:],yp[mask]) #train with all labeled data
	if method =='svm-base':
            counter=0
            before=np.ones(yp.shape)
            a=2
            b=1
            while True:
		model = svm.NuSVC(kernel='sigmoid')
                model.fit(X[mask,:],yp[mask])
                if b == a or counter == self.max_iter or a==0:
                    break

                probs = model.predict_proba(X[~mask])

                b=np.sum(~mask)

                yp[~mask] = np.fromiter([getLabel(p) for p in probs], probs.dtype)

                counter+=1
                mask = np.ones(len(y), dtype=bool)
                mask[np.where(yp==-1)[0]]=False
                a=np.sum(~mask)  # number of yp==-1 true number of unlabelled data

        elif method=='self-training':
            counter=0
            before=np.ones(yp.shape)
            a=2
            b=1
            while True:
                lda.fit(X[mask,:],yp[mask])
                if b == a or counter == self.max_iter or a==0:
                    break
                #print(X[~mask])  #print labelled data
                probs = lda.predict_proba(X[~mask])
                #print(probs)
                b=np.sum(~mask)
                #print(b)
                yp[~mask] = np.fromiter([getLabel(p) for p in probs], probs.dtype)
                #print(yp)
                counter+=1
                mask = np.ones(len(y), dtype=bool)
                mask[np.where(yp==-1)[0]]=False
                a=np.sum(~mask)  # number of yp==-1 true number of unlabelled data
                #print(a)
            #print(counter) 
        elif method == 'label-propagation':
            counter =0
            b=1000
            a=10001
            while True:
                
                if b == a or counter == self.max_iter or a==0:
                    break
                label_prop_model=LabelPropagation(kernel='rbf',n_neighbors=10,alpha=0.9)
                label_prop_model.fit(X,yp)
                probs = label_prop_model.predict_proba(X[~mask])
                
                yp[~mask] = np.fromiter([getLabel(p) for p in probs], probs.dtype)
                
                self.propagated_labels = yp
                b=np.sum(~mask)
                mask = np.ones(len(y),dtype=bool) #mask of labeled data
                mask[np.where(yp==-1)[0]] = False #cheke unlabeled data , msk = number of labeled data
                counter+=1
                a=np.sum(~mask)

            lda.fit(X[mask,:],yp[mask])

        
        else:
            raise('No valid method was given!')
        self.classifier, self.means_, self.covariance_ =lda, lda.means_, lda.covariance_
