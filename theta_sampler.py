#Simulates a learner who has inferred transitivity categories for each verb in data
	#and noise parameters (epsilon and delta) from previous rounds of Gibbs sampling
#Samples values for theta (direct object rates) for each verb that was inferred to be alternating
#Runs 1000 steps of Metropolis-Hastings simulation using MH_theta.py
	#to sample from distribution over theta for each verb
#Data: a list of length n where each item is a 2-element list corresponding 
#   to counts of observations for each of n alternating verbs. In each sublist, the first element 
#   contains counts of direct objects and the second contains total number of observations
#Epsilon and delta: point estimates of epsilon and delta, a decimal between 0 and 1
#Returns an nxv matrix of sampled values of theta for each of v verbs, for each of n iterations

import math
import random
import numpy as np
from MH_theta import MH_theta


def sample_theta(data, delta, epsilon):
	
	gammas = {}
	thetas = MH_theta(data, delta, epsilon, gammas, 1000)
	
	#Use every 10th value from last 500 iterations as samples
	thetasamples = thetas[501::10]

	#Saves table with each column corresponding to samples from a single verb
	thetatable = np.asarray(thetasamples)
	np.savetxt('thetatable.csv', thetatable, fmt='%1.2f', delimiter=",")
	
	return thetatable

#data containing vector of verb counts from CHILDES Treebank
#first list element for each verb in the vector is counts of overt direct objects,
#and second list element is total count for that verb.
#see 'CHILDESTreebank_VerbData' for all 50 verbs in order.
data = [[308,1568], [777,1318], [11,859], [541,605], [3,605], [155,583], [406,579], [347,550], [350,509], [350,485], [287,477], [13,451], [57,383], [193,375], [221,366], [299,358], [297,356], [274,352], [265,342], [305,337], [299,331], [268,331], [275,312], [4,308], [161,306], [215,299], [21,294], [114,281], [8,275], [198,263], [11,256], [132,255], [11,253], [112,238], [13,228], [49,227], [205,220], [187,214], [8,197], [161,195], [57,192], [140,191], [141,185], [160,185], [153,183], [7,180], [149,169], [141,166], [115,160], [53,151]]

print(sample_theta(data, 0.21, 0.11))
