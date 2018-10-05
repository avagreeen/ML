import numpy as np
from random import choice
#np.seterr(divide='ignore', invalid='ignore')
Q_left = np.zeros(6)
Q_right = np.zeros(6)

Q_i_left = [0, 1, 0.5, 0.625, 1.25, 0];
Q_i_right = [0, 0.625, 1.25, 2.5, 5, 0];
d=np.zeros(10000)
#delta=[]
#phi=np.zeros(6)
#theta_left = np.ones(6)
#theta_right = np.ones(6)
width = 1.5
e=0.9
alpha=0.1
gamma=0.5

def State(s,a):
	if (np.random.uniform(0,1) > 0.3):
		s_p = s+a
	else:
		s_p=s
	return s_p
def rbf(s,width):
    c=range(1,7)
    #print s,c
    dis=s-np.array(c)
  #  print 'dis'+str(dis)
    phi=np.exp(-dis**2/(2*width**2))
 #   print 'phi'+str(phi)
    return phi/sum(phi)   
#explore next state and update phi with current s
'''
def e_greedy(e,s,width):
    if(np.random.uniform(0,1) > e): # with prob 1-e pick the greedy action
        phii = rbf(s,width)
        #print phii
        left = np.dot(theta_left,phii) +1
        right = np.dot(theta_right,phii)
        a = np.sign(right - left)
    else:
        a = choice([-1,1])
    s_p=State(s,a)
    
    return phii,left,right
'''


def Qmax(s_p,width,theta_left,theta_right):
    
    p=rbf(s_p,width)
    n_left=np.dot(theta_left,p)
    n_right=np.dot(theta_right,p)
    return max(n_left,n_right)
    
def Reward(s,a):
    r=0
    if (s<1.5):
        r=1
    if (s>=5.5):
        r=5
    return r
def theta_update(theta, delta,alpha,phi):
    theta_new = theta + alpha*delta*phi
    return theta_new

def q_learn(e,width,Q_left,Q_right,Q_i_left,Q_i_right,alpha,gamma):
    #global phi
    #global delta
    #global theta_left,theta_right
    iteration =10000
    delta = []
    phi=np.zeros(6)
    theta_left = np.ones(6)
    theta_right = np.ones(6)
    s = choice([2,3,4,5])
    d=np.zeros([iteration])
    for i in range(iteration):
         #init state
        #while (s>=1.5 and s<5.5):
        if(np.random.uniform(0,1) > e): # with prob 1-e pick the greedy action
                
            phi = rbf(s,width)
            #print phii
            left = np.dot(theta_left,phi) +1
            right = np.dot(theta_right,phi)
            a = np.sign(right - left)
        else:
            a = choice([-1,1])
        s_p=State(s,a)
        
        R=Reward(s_p,a)
        q=Qmax(s_p,width,theta_left,theta_right)
        q_next = R+gamma*q
        #print '***'
        #print s_p
            #print phi
        if a == -1:
            #delta = q_next - np.dot(rbf(s_p,width),theta_left)
            #theta_left = theta_update(theta_left,delta,alpha,phi)
            theta_left = theta_left+alpha*(q_next-np.dot(phi,theta_left))*phi
                #print 'left'+ str(theta_left)
        if a == 1:
            #print 'right'
            #delta = q_next - np.dot(rbf(s_p,width),theta_right)
            #theta_right = theta_update(theta_right,delta,alpha,phi)
            theta_right = theta_right+alpha*(q_next-np.dot(phi,theta_right))*phi
                #print 'right'+ str(theta_right)
        s=s_p
        #print '**********************' + str(s)
        #print 'theta_left ' + str(theta_left)
       # print 'theta_right ' + str(theta_right)

    for j in xrange(1,5):
        Q_left[j] = np.dot(rbf(j,width),theta_left)
        
        
        #print Q_i_left
        Q_right[j] = np.dot(rbf(j,width),theta_right)
        #print Q_right
    if (s<1.5 or s>=5.5):
        s=choice([2,3,4,5])
    
    d[i]=np.sqrt(np.sum((Q_i_left-Q_left)**2+np.sum(Q_i_right-Q_right)**2))
    return d
 #   print Q_left
d=q_learn(e,width,Q_left,Q_right,Q_i_left,Q_i_right,alpha,gamma)
import matplotlib.pyplot as plt

plt.figure
plt.plot(d)
plt.show()