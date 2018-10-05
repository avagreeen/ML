import numpy as np
from random import choice
#np.seterr(divide='ignore', invalid='ignore')
Q_left = np.zeros(6)
Q_right = np.zeros(6)

Q_i_left = [0, 1, 0.5, 0.625, 1.25, 0];
Q_i_right = [0, 0.625, 1.25, 2.5, 5, 0];

width = 1
e=0.9
alpha=0.001
gamma=0.5
#



def State(s,a):
    
    return s+a+np.random.normal(0,0.01)

def rbf(s,width):
    c=range(1,7)
    #c=np.array([1.5,5.5])
    dis=s-np.array(c)
    phi=np.exp(-dis**2/(width**2))
    return phi/sum(phi)   
#explore next state and update phi with current s

def Qmax(s_p,width,theta_left,theta_right):
    
    #p=rbf(s_p,width)
    n_left=np.dot(theta_left,rbf(State(s_p,-1),width))
    n_right=np.dot(theta_right,rbf(State(s_p,1),width))
    return max(n_left,n_right)
    
def Reward(s):
    if (s<1.5):
        r=1
    if (s>=5.5):
        r=5
    else:
        r=0
    return r
def theta_update(theta, delta,alpha,phi):
    theta_new = theta + alpha*delta*phi
    return theta_new
def greedy(e,s,theta_left,theta_right):
    if(np.random.uniform(0,1) >e): # with prob 1-e pick the greedy action
        s_left=State(s,-1)
        q_s_left=np.dot(rbf(s_left,width),theta_left)
        s_right=State(s,1)
        q_s_right=np.dot(rbf(s_right,width),theta_right)
        if q_s_left > q_s_right:
            s_p=s_left
            a=-1
        else:
            s_p=s_right
            a=1
    else:
        a=choice([-1,1])
        s_p=State(s,a)
    print a,s_p
    return int(a),s_p
def q_learn(e,width,Q_left,Q_right,Q_i_left,Q_i_right,alpha,gamma):
    delta = []
    iteration=22000
    d=np.zeros(iteration)
    num_phi=6
    phi=np.zeros(num_phi)
    theta_left = np.ones(num_phi)
    theta_right = np.ones(num_phi)
    for i in range(iteration):
        s = choice([2,3,4,5]) #init state
        while (s>=1.5 and s<5.5):
            
            if(np.random.uniform(0,1) > e): # with prob 1-e pick the greedy action
                #print phii
                left = np.dot(theta_left,rbf(s-1+np.random.normal(0,0.01),width))
                right = np.dot(theta_right,rbf(s+1+np.random.normal(0,0.01),width))
                a = int(np.sign(right - left))
            else:
                a = choice([-1,1])
            s_p=State(s,a)
            #a,s_p= greedy(e,s,theta_left,theta_right)
            
            R=Reward(s_p)
            q=Qmax(s_p,width,theta_left,theta_right)

            q_next = R+gamma*q
            #alpha=1/(i+100)
            phi = rbf(s,width)
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
            #print theta_left
            Q_left[j] = np.dot(rbf(j,width),theta_left)
        #print Q_i_left
            Q_right[j] = np.dot(rbf(j,width),theta_right)
        #print Q_right'

        if(s<1.5 || s>=5.5):
            s = 4;
       
          
        #d[i]=np.sqrt(np.sum((Q_i_left-Q_left)**2+np.sum(Q_i_right-Q_right)**2))
        d[i]=np.sqrt(sum((Q_left-Q_i_left)**2)+sum((Q_right - Q_i_right)**2))
    print theta_left
    print theta_right
        
    print Q_left
    print Q_right
    return d,theta_left,theta_right
#%%
d=q_learn(e,width,Q_left,Q_right,Q_i_left,Q_i_right,alpha,gamma)
#%%
Width=[0.1,0.5,1,1.5,2.5,3.5]
import matplotlib.pyplot as plt

for i,width in enumerate(Width):
    #for j,alpha in enumerate(Alpha):
    diff,theta_left,theta_right=q_learn(e,width,Q_left,Q_right,Q_i_left,Q_i_right,alpha,gamma)
    plt.figure
    plt.plot(diff)
    
plt.legend(['width=0.1','width=0.5','width= 1','width= 1.5','width=2.5','width=3.5'])
#
plt.ylabel('2-norm difference')
plt.xlabel('iterations')
plt.title('6 base function centered')
plt.savefig('width_6_base',dpi=140)
plt.show()

#%%
xaxis=np.linspace(0.5,6.5,100)
y1=np.exp(-(xaxis-1)**2/(1**2*2))
y2=np.exp(-(xaxis-5)**2/(1**2*2))



plt.plot(xaxis,yaxis)