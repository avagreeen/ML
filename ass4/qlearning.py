import numpy as np
from random import choice

Q_left = np.zeros(6)
Q_right = np.zeros(6)

Q_f_left = [0, 1, 0.5, 0.625, 1.25, 0];
Q_f_right = [0, 0.625, 1.25, 2.5, 5, 0];

action=[-1,1]
gamma=0.5

Epsilon=[0.01, 0.1, 0.2, 0.4, 0.5,0.9,1]
Alpha = [0.01,0.1, 0.2, 0.4, 0.5,0.9,1]
reward = [1,0,0,0,0,5]

#e-greedy decide the action This 
#procedure is adopted to minimize the possibility of overfitting during evaluation.
def e_greedy(e,left,right):
	if(np.random.uniform(0,1) > e): # with prob 1-e pick the greedy action
		a = np.sign(right - left)
	else:
		a = choice(action)
	return int(a)

def Q_learning(epsilon,alpha):
    diff=[]
    for i in xrange(1000):
		s = choice([1,2,3,4]) #init state
		while (s!=0 and s!=5):
			a = e_greedy(epsilon,Q_left[s],Q_right[s])
			s_p = s+a #get estimate step
		
			R = reward[s_p];
			if a == -1:
				Q_left[s]=Q_left[s]+alpha*(R+gamma*max(Q_left[s_p],Q_right[s_p])-Q_left[s])
			if a == 1:
				Q_right[s]=Q_right[s]+alpha*(R+gamma*max(Q_left[s_p],Q_right[s_p])-Q_right[s])
			s=s_p
            diff[i] = np.linalg.norm(Q_f_left-Q_left)+np.linalg.norm(Q_f_right-Q_right)
	
	return Q_left,Q_right,diff

#print Q_learning(0.3, 0.1)
# compute 2 norm diff
#for i,alpha in enumerate(Alpha):
    #%%
import matplotlib.pyplot as plt
diff=np.zeros([len(Epsilon),len(Alpha)])
for i,epsilon in enumerate(Epsilon):
	for j,alpha in enumerate(Alpha):
		 Q_left,Q_right,diff=Q_learning(epsilon,alpha)
		 #diff[i,j] = sum((Q_left-Q_f_left)**2)+sum((Q_right - Q_f_right)**2)
		 #print alpha,epsilon
		 #print(Q_left,Q_right)

#diff = sum((Q_left-Q_f_left)**2+(Q_right - Q_f_right)**2)


#Q_left,Q_right,diff=Q_learning(0.1,0.5)
#diff = np.linalg.norm(Q_f_left-Q_left)+np.linalg.norm(Q_f_right-Q_right)
#print diff
