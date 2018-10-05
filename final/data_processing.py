import matplotlib.pyplot as plt
import numpy as np


#%%
import csv
data=[]
with open('train.csv') as f:
    reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        data.append(row)
np_data=np.array(data)
#%%
label= np_data[:,0]
feature = np_data[:,1:]
#%%
from sklearn import svm
