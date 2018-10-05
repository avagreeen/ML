import numpy as np
import numpy.random as rnd
import math as m
from scipy.stats import multivariate_normal, bernoulli
from copy import copy
from sklearn import preprocessing as pp
import csv
import numpy as np
from sklearn import datasets
import random
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

def load_file(filename):
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    data = np.array(list(reader))
    X = data[:,:len(data[0]) - 1]
    X = np.array(X).astype('float')
    Y = data[:, -1]
    return (X,Y)

def load_magic():
    X, y = load_file('magic04.data')

    for i in range(len(y)):
        if y[i] == 'g':
            y[i] = 0
        elif y[i] == 'h':
            y[i] = 1
    
#delete duplicates in nparray
    y=y.astype(np.int64)
    y=y.reshape((len(y),1))
    data=np.hstack((X,y))
    unique_index= np.unique(data.dot(np.random.rand(11)), return_index= True)[1]
    u_data=data[unique_index]
    X = u_data[:,range(10)]
    y=u_data[:,-1]
    X=pp.scale(X)
    return (X, y)

def split_data(X,y,N_unlabeled):
    #get all the training data
    items = random.sample(range(len(y)),25+N_unlabeled)
    X_train = X[items,]
    y_train_true=y[items,]
    u_items = random.sample(range(len(y_train_true)),N_unlabeled)
    
    y_train = copy(y_train_true)
    train_mask_unl = np.zeros(y_train_true.shape, dtype=bool)
    train_mask_unl[u_items] = True
    y_train[train_mask_unl] = -1
    
    #get test data
    train_mask = np.zeros(X.shape[0], dtype=bool)
    train_mask[items]=True
    X_test,y_test=X[~train_mask,:],y[~train_mask]
    #print(y_train_true)
    #print('*************************')
    return X_train, y_train, y_train_true, X_test, y_test

def genData():
    X, y = make_circles(n_samples=1000, noise=0.1)
    # scatter plot, dots colored by class value
    '''
    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    colors = {0:'red', 1:'blue'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    #pyplot.show()
    '''
    return X,y

def genGaussion():
    X, y = make_blobs(n_samples=1000, centers=[(0,0), (2,2.5)], cluster_std=2)

    return X,y

def genMoons():
    X, y = make_moons(n_samples=1000, noise=0.1)
    return X,y
def genReg():
    X1, X2 = datasets.make_regression(n_samples=500,n_features=1,n_targets=1,noise=40)
    y1=np.ones(len(X1))
    #plt.scatter(X1,X2)
    X3, X4 = datasets.make_regression(n_samples=500,n_features=1,n_targets=1,noise=40)
    X3=X3+1
    #plt.scatter(X3+2,X4)
    y2=np.zeros(len(X3))
    Xp=np.hstack((X1,np.reshape(X2,(500,1))))
    Xn=np.hstack((X3,np.reshape(X4,(500,1))))
    y=np.vstack((y1,y2))
    labels=np.reshape(y,(1000,))
    labels=labels.astype(np.int64)
    X=np.vstack((Xp,Xn))
    return X,labels

def gen_biased_train_data():
    xp1=np.reshape(np.random.sample(5)*10-20,(5,1)) #blue
    xp2=np.reshape(np.random.sample(5)*10-200,(5,1))
   # plt.scatter(xp1,xp2)
    xn1=np.reshape(np.random.sample(20)*10+20,(20,1))#red
    xn2=np.reshape(np.random.sample(20)*10+200,(20,1))
   # plt.scatter(xn1,xn2)
#

    Xtp=np.concatenate((xp1,xp2),axis=1)
    Xtn=np.hstack((xn1,xn2))
    Xt=np.vstack((Xtp,Xtn))

    ytp=np.reshape(np.ones(len(xp1)),(5,1))
    ytn=np.reshape(np.zeros(len(xn1)),(20,1))

    yt=np.reshape(np.vstack((ytp,ytn)),(25,1))
    yt=yt.astype(np.int64)
    #test data
    X1, X2 = datasets.make_regression(n_samples=500,n_features=1,n_targets=1,noise=20)
    y1=np.ones(len(X1))
    #plt.scatter(X1,X2)
    X3, X4 = datasets.make_regression(n_samples=500,n_features=1,n_targets=1,noise=20)
    #plt.scatter(X3+2,X4)
    y2=np.zeros(len(X3))
    X2=np.reshape(X2,(500,1))
    X4=np.reshape(X4,(500,1))

    Xp=np.hstack((X1*10,X2-7*X1-100))
    Xn=np.hstack((X3*10+2,X4-7*X3+100))

    y=np.vstack((y1,y2))
    labels=np.reshape(y,(1000,1))
    labels=labels.astype(np.int64)
    X=np.vstack((Xp,Xn))
    #concatinate all the data
    X=np.concatenate((Xt,X),axis=0)
    y=np.concatenate((yt,labels),axis=0)
    y=np.reshape(y,(len(y),))
    return X,y
def split_customed_data(X,y,N_unlabeled):
    items = random.sample(range(25,len(y)),N_unlabeled)
    X_train = np.concatenate((X[range(25),],X[items,]),axis=0)
    #print(X_train.shape)
    y_train_true= np.concatenate((y[range(25),],y[items,]),axis=0)
    y_train=copy(y_train_true)
    y_train[25:-1] = -1
    
    #get test data
    train_mask = np.zeros(X.shape[0], dtype=bool)
    train_mask[items]=True
    X_test,y_test=X[~train_mask,:],y[~train_mask]
    #print(y_train_true)
    #print('*************************')
    return X_train, y_train, y_train_true, X_test, y_test

def gen_biased_train_data1():
    xp1=np.reshape(np.random.sample(5)+2,(5,1))
    xp2=np.reshape(np.random.sample(5),(5,1))
   # plt.scatter(xp1,xp2)
    xn1=np.reshape(np.random.sample(20)-1,(20,1))
    xn2=np.reshape(np.random.sample(20),(20,1))
   # plt.scatter(xn1,xn2)
#

    Xtp=np.concatenate((xp1,xp2),axis=1)
    Xtn=np.hstack((xn1,xn2))
    Xt=np.vstack((Xtp,Xtn))

    ytp=np.reshape(np.ones(len(xp1)),(5,1))
    ytn=np.reshape(np.zeros(len(xn1)),(20,1))

    yt=np.reshape(np.vstack((ytp,ytn)),(25,1))
    yt=yt.astype(np.int64)
    #test data
    X, y = make_moons(n_samples=1000, noise=0.3)


    #concatinate all the data
    X=np.concatenate((Xt,X),axis=0)
    y=np.concatenate((yt,np.reshape(y,(len(y),1))),axis=0)
    y=np.reshape(y,(len(y),))
    return X,y
def gen_bias_moon():
    X,y=genMoons()
    #y=np.reshape(y,(1000,1))
    X1=copy(X)
    X2=copy(X)
    for i in range(len(y)):
        if y[i]==0:
            X[i,0]=1.5+X[i,0]
            X[i,1]-=1
        if y[i]==1:
            X[i,0]=X[i,0]-1.5
        X[i,1]=X[i,1]*5
    #X11=X[:,0]+1
    #print(X.shape)
    
    xp1=np.reshape(np.random.sample(5)-0.5,(5,1)) #blue
    xp2=np.reshape(np.random.sample(5)-0.6,(5,1))
   # plt.scatter(xp1,xp2)
    xn1=np.reshape(np.random.sample(20)+1,(20,1))#red
    xn2=np.reshape(np.random.sample(20)-0.5,(20,1))
   # plt.scatter(xn1,xn2)
#

    Xtp=np.concatenate((xp1,xp2),axis=1)
    Xtn=np.hstack((xn1,xn2))
    Xt=np.vstack((Xtp,Xtn))

    ytp=np.reshape(np.ones(len(xp1)),(5,1))
    ytn=np.reshape(np.zeros(len(xn1)),(20,1))

    yt=np.reshape(np.vstack((ytp,ytn)),(25,1))
    yt=yt.astype(np.int64)
#
    X=np.concatenate((Xt,X),axis=0)
    y=np.concatenate((yt,np.reshape(y,(len(y),1))),axis=0)
    y=np.reshape(y,(len(y),))
    
    return X,y

