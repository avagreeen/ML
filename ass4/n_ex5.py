import numpy as np
from random import choice
import matplotlib.pyplot as plt

Q_left = np.zeros(6)
Q_right = np.zeros(6)
iteration=4000
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

def State(s,a):
	if (np.random.uniform(0,1) > 0.3):
		s_p = s+a
	else:
		s_p=s
	return s_p
'''
def Q_learning(epsilon,alpha):
	for i in range(1000):
		s = choice([1,2,3,4]) #init state
		while (s!=0 and s!=5):
			a = e_greedy(epsilon,Q_left[s],Q_right[s])
			s_p = State(s,a) #get estimate step
			R = reward[s_p];
			#print s,a

			if a == -1:
				Q_left[s] = Q_left[s]+alpha*(R+gamma*max(Q_left[s_p],Q_right[s_p])-Q_left[s])
			if a == 1:
				Q_right[s] = Q_right[s]+alpha*(R+gamma*max(Q_left[s_p],Q_right[s_p])-Q_right[s])
			s=s_p

	return Q_left,Q_right
'''
def Q_learning_broken(epsilon,alpha):
	global iteration
	 #init state
	Q_left = np.zeros(6)
	Q_right = np.zeros(6)
	diff=np.zeros(len(range(iteration)))
	for i in xrange(iteration):
		s = choice([1,2,3,4])
		while (s!=0 and s!=5):
			a = e_greedy(epsilon,Q_left[s],Q_right[s])
			s_p = State(s,a) #get estimate step
			R = reward[s_p];
			#print s,a
			if a == -1:
				Q_left[s] = (1-alpha)*Q_left[s]+(alpha*(R+gamma*max(Q_left[s_p],Q_right[s_p])))
			if a == 1:
				Q_right[s] = (1-alpha)*Q_right[s]+(alpha*(R+gamma*max(Q_left[s_p],Q_right[s_p])))
			s=s_p
		diff[i] = np.sqrt(sum((Q_left-Q_f_left)**2)+sum((Q_right - Q_f_right)**2))

	#plt.plot(diff)
	#print Q_left,Q_right
	#print 'q_learning'
	return diff
def Q_iter(alpha):
	for i in xrange(10000):
		for s in xrange(1,5):
			for a in action:
				s_p = State(s,a)
				#print a
				if a == -1:
					Q_left[s] = reward[s_p]+gamma*max(Q_left[s_p],Q_right[s_p])
					Q_right[s] = reward[s_p]+gamma*max(Q_left[s_p],Q_right[s_p])
		diff[i] = np.sqrt(sum((Q_left-Q_f_left)**2)+sum((Q_right - Q_f_right)**2)

#	return diff

#%%
#%%
'''
diff = Q_learning_broken(0.7,0.2)
plt.plot(diff)
diff = Q_learning_broken(0.9,0.1)
plt.plot(diff)
'''
diff = Q_learning_broken(0.9,0.01)
plt.plot(diff)
plt.legend(['epsilon=0.7,alpha=0.2', 'epsilon=0.9,alpha=0.1', 'epsilon=0.9,alpha=0.01'])
plt.savefig('ex5')
plt.show()
#Q_iter()
#%%
diff = Q_iter(0.1)
plt.plot(diff)
diff = Q_iter(0.3)
plt.plot(diff)
diff = Q_iter(0.5)
plt.plot(diff)
plt.legend(['epsilon=0.7,alpha=0.2', 'epsilon=0.9,alpha=0.1', 'epsilon=0.9,alpha=0.01'])
plt.savefig('ex5')
plt.show()
