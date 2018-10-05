#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 15:03:56 2018

@author: xin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROBA = 0.3
N_STATES = 6  # 1维世界的宽度
ACTIONS = ['left', 'right']  # 探索者的可用动作
EPSILON = 0.8
ALPHA = 0.6  # 学习率
GAMMA = 0.5  # 奖励递减值
MAX_EPISODES = 2000 # 最大回合数
FRESH_TIME = 0.3    # 移动间隔时间

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table 全 0 初始
        columns=actions,  # columns 对应的是行为名称
    )
    return table


def state_change(S, A):
    mu = 0
    sigma = 0.01
    G = np.random.normal(mu, sigma, 1) # Gaussian noise 
    if S >= 0.5 and S < 4.5:
        if A == 'right':    # move right
            S_ = S + 1 + G
        else:   # move left
            S_ = S - 1 + G
    else:
        S_ = S
        
    return S_

   


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    # with probability the agent will stay in the same location
    mu = 0
    sigma = 0.01
    G = np.random.normal(mu, sigma, 1) # Gaussian noise   
    if S >= 0.5 and S < 4.5:
        if A == 'right':    # move right
            S_ = S + 1 + G
            if S_ >= 4.5:   # terminate
               #  S_ = 'terminal'
                R = 5
            else:               
                R = 0
        else:   # move left
            S_ = S - 1 + G
            if S_ < 0.5:
                # S_= 'terminal' # terminate
                R = 1
            else:               
                R = 0
    else:
        S_ = S
        R = 0
    return S_, R

# center point of rbf
center = [0, 1, 2, 3, 4, 5]

# expected return
expected = np.array([[0.0, 1.0, 0.5, 0.625, 1.25, 0.0],
                [0.0, 0.625, 1.25, 2.5, 5.0, 0.0]])

Q_expected = pd.DataFrame(
    expected.T,  
    columns=ACTIONS,  # columns 对应的是行为名称
)


def rbf(x, c, s):
    if s != 0:
        return np.exp(-1 / (2 * s**2) * (x-c)**2)
    else:
        if x == c:
            return 1
        else:
            return 0

def RBF_values(S):
    std = 0.1 # change std
    # center point of rbf
    center = [0, 1, 2, 3, 4, 5]
    rbf_values = np.zeros(6)
    for i in range(N_STATES):
        rbf_values[i] = rbf(S,center[i],std)
    
    return rbf_values

def normalization(x):  
    if( np.max(x) - np.min(x)) != 0:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))  
    return x  


def choose_action(state, EPSILON, theta):
    q_left = np.dot(RBF_values(np.asscalar(state_change(state, 'left')[0])).T, theta.loc[:,'left'])
    q_right = np.dot(RBF_values(np.asscalar(state_change(state, 'right')[0])).T, theta.loc[:,'right'])
    state_actions = np.array([q_left,q_right]) # 选出这个 state 的所有 action 值
    if (np.random.uniform() < EPSILON) or (np.sum(state_actions) == 0):  # 非贪婪 or 或者这个 state 还没有探索过
        action_name = np.random.choice(ACTIONS)
        #print('random')
    else:
        if q_left < q_right:
            action_name =  'right'  # 贪婪模式
        else:
            action_name =  'left' 
    return action_name

def rbf_ql():
    # diff = 0.001
    MAX_EPISODES = 500
    # initialize weight 
    theta_array = np.zeros((6,2))
    
    theta = pd.DataFrame(
        theta_array,  
        columns=ACTIONS,  # columns 对应的是行为名称
    )
    # Q_predict = build_q_table(N_STATES, ACTIONS)  # 初始 q table
#    # expected return
    real = np.array([[0.0, 1.0, 0.5, 0.625, 1.25, 0.0],
                    [0.0, 0.625, 1.25, 2.5, 5.0, 0.0]]).T
    
#    Q_real = pd.DataFrame(
#        real.T,  
#        columns=ACTIONS,  # columns 对应的是行为名称
#    )
    
    step = 0.05
    theta_result = []
    diff_q = []
    result =[]
    for episode in range(MAX_EPISODES):     # 回合
        S = np.random.randint(1, 4)  # 回合初始位置
        is_terminated = False   # 是否回合结束
        rbf_values = np.zeros(N_STATES)
        while not is_terminated:
            A = choose_action(S, EPSILON, theta)   # 选行为
            S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈
            S_ = float(S_)
           # type(S_)
            rbf_values = RBF_values(S)
            Q_predict = np.dot(theta.loc[:,A], rbf_values)
            if S >= 0.5 and S < 4.5:
                
                q_left = np.dot(RBF_values(S_), theta.loc[:,'left'])
                q_right = np.dot(RBF_values(S_), theta.loc[:,'right']) 
#            else:
#                q_left = q_right  = 0
                
            
#            if S_ >= 0.5 and S_ < 4.5:
#                q_left = np.dot(RBF_values(float(state_change(S_, 'left'))).T, theta.loc[:,'left'])
#                q_right = np.dot(RBF_values(float(state_change(S_, 'right'))).T, theta.loc[:,'right'])
#            else:
#                q_left = np.dot(RBF_values(float(state_change(S_, 'left'))).T, theta.loc[:,'left'])
#                q_right = np.dot(RBF_values(float(state_change(S_, 'right'))).T, theta.loc[:,'right'])
#                # q_left = q_right = 0
#                is_terminated = True # terminate and begin again
            Q_expected = R + GAMMA * max(q_left, q_right)
            # L = 1/2.0 * np.sqrt(Q_predict.iloc[S,A] - Q_expected.iloc[S,A])
            delta = Q_expected - Q_predict
            
            if A == 'left':               
                theta.loc[:,'left'] = theta.loc[:,'left']  + step * delta * rbf_values
            else:
                theta.loc[:,'right'] = theta.loc[:,'right']  + step * delta * rbf_values
                
            theta.loc[:,'right']  = theta.loc[:,'right'] #/sum(theta.loc[:,'right'] )
            theta.loc[:,'left']  = theta.loc[:,'left'] #/ sum(theta.loc[:,'left'] )
            S = S_  # 探索者移动到下一个 state
            if S_ < 0.5 or S_ >= 4.5:
                S = np.random.randint(1, 4)  # choose state randomly
                is_terminated = True # terminate and begin again
        theta_result.append(np.sum(theta))
        q_table = np.zeros((6,2))
        for i in range(0,6):
            # print(i)
            q_table[i,0] = np.dot(RBF_values(i),  theta.loc[:,'left'])
            q_table[i,1] = np.dot(RBF_values(i),  theta.loc[:,'right'])
        print(q_table)
        result.append(np.linalg.norm(q_table - real))
    diff_q.append(result)
        
#        if delta < diff:
#            break
        

    return theta, q_table,  theta_result, diff_q
    #return Q_predict


if __name__ == "__main__":
    theta, q_table, theta_result, diff_q = rbf_ql()
    print(theta)
    plt.plot(theta_result)
    plt.legend(['left_theta','right_theta'])
    plt.show()
    for i in range(500):
        plt.plot(np.array(diff_q)[i])
    plt.ylabel('2-norm differnce')
    plt.show()

    Q_expected = pd.DataFrame(
                q_table,  
                columns=ACTIONS,  # columns 对应的是行为名称
                )
    print(Q_expected)

#state = 0.5:0.2:6.5;
#state_left = zeros(1, length(state));
#state_right = zeros(1, length(state));
#Q_f_left = zeros(1, length(state));
#Q_f_right =  zeros(1, length(state));
#for j = 6:length(state)-5
#    state_left(j) = State_N(state(j), -1);
#    state_right(j) = State_N(state(j), 1);
#    Q_f_left(j) = RBF(state_left(j), rho)'*theta(1, :)';
#    Q_f_right(j) = RBF(state_right(j), rho)'*theta(2, :)';
#end