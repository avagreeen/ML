import numpy as np
from random import choice
import pdb
import crash_on_ipy

Q_left = np.zeros(6)
Q_right = np.zeros(6)

action=[-1,1]
gamma=0.5

reward = [1,0,0,0,0,5]

for i in xrange(100):
	for s in xrange(1,5):
		print(s)
		for a in action:
			s_p = s+a
			print a
			if a == -1:
				Q_left[s] = reward[s_p]+gamma*max(Q_left[s_p],Q_right[s_p])
			if a == 1:
				Q_right[s] = reward[s_p]+gamma*max(Q_left[s_p],Q_right[s_p])
print Q_left,Q_right