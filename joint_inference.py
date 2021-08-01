#Simulates a learner encountering a corpus of observations of many verbs (data)
#Infers model values (aka, verb transitivity classes) for each verb in data:
    #1: verb is fully transitive (theta = 1)
    #2: verb is fully intransitive (theta = 0)
    #3: verb is mixed (theta sampled from Beta(1,1) uniform distribution
#Infers a single value of epsilon and delta across all verbs in
    #data, a decimal between 0 and 1
#At each iteration, runs 10 steps of the Metropolis-Hastings simulation in
    #MH_epsilon and MH_delta, and uses the 10th values generated to sample model
    #values using sample_models.m
#Data: a list of length n where each item is a 2-element list corresponding 
#   to counts of observations for each of n verbs. In each sublist, the first element 
#   contains counts of direct objects and the second contains total number of observations
#Iterations: number of iterations to run simulation, must be an integer value
#Returns epsilon, a list of length n of epsilon values, delta, a list of length n of delta values,
	#and models, an nxv matrix of model values for each of v verbs, for each of n iterations

import math
import random
from sample_models import sample_models
from MH import MH

def joint_inference(data, iterations):

	#Randomly initialize epsilon and delta
	epsilon = [random.random()]
	delta = [random.random()]
	models = []
	gammas = {}
	
	for i in range(0, iterations):
	
		print('iteration', i)
		
		#Use current epsilon and delta to infer model values
		newmodels = sample_models(data, epsilon[i], delta[i], gammas)
		print('models', newmodels)
		models.append(newmodels)
		
		#Run Metropolis-Hastings simulation 10 times to infer new epsilon
		#from current delta and model values
		#MH sampling on epsilon, so boolean value is False
		timelogepsilon = MH(data, models[i], delta[i], epsilon[i], gammas, 10, False)
		newepsilon = timelogepsilon[9]
		epsilon.append(newepsilon)
		
		#Run Metropolis-Hastings simulation 10 times to infer new delta
		#from new epsilon and model values
		#MH sampling on delta, so boolean value is True
		timelogdelta = MH(data, models[i], delta[i], newepsilon, gammas, 10, True)
		newdelta = timelogdelta[9]
		delta.append(newdelta)
	
	return models, epsilon, delta

#Run joint_inference over 1000 iterations and plot probability distribution over
#models and epsilon

import numpy as np
import matplotlib.pyplot as plt

def plot_joint_inference(data):
	
	models, epsilon, delta = joint_inference(data, 1000)
	
	#Use every 10th value from last 500 iterations as samples
	modelsamples = models[501::10]
	epsilonsamples = epsilon[501::10]
	np.savetxt('epsilon', epsilonsamples)
	deltasamples = delta[501::10]
	np.savetxt('delta', deltasamples)
	
	#Plot histogram of epsilon samples
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title('Distribution over Epsilon')
	ax.hist(epsilonsamples)
	fig.savefig('epsilon.png')
	
	#Plot histogram of delta samples
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title('Distribution over Delta')
	ax.hist(deltasamples)
	fig.savefig('delta.png')
	
	#Display table containing counts of model assignments per model for each verb
	modelstransposed = list(map(list, zip(*modelsamples)))
	modeltable = []
	
	for i in range(0, len(data)):
		histcounts = np.histogram(modelstransposed[i], bins = [1, 2, 3, 4])
		modeltable.append(histcounts[0])
	
	modeltable = np.asarray(modeltable)
	np.savetxt('modeltable', modeltable, fmt='%1i')
	
	return modeltable

#data containing vector of verb counts from CHILDES Treebank
#first list element for each verb in the vector is counts of overt direct objects,
#and second list element is total count for that verb.
#see 'CHILDESTreebank_VerbData' for all 50 verbs in order.
data = [[308,1568], [777,1318], [11,859], [541,605], [3,605], [155,583], [406,579], [347,550], [350,509], [350,485], [287,477], [13,451], [57,383], [193,375], [221,366], [299,358], [297,356], [274,352], [265,342], [305,337], [299,331], [268,331], [275,312], [4,308], [161,306], [215,299], [21,294], [114,281], [8,275], [198,263], [11,256], [132,255], [11,253], [112,238], [13,228], [49,227], [205,220], [187,214], [8,197], [161,195], [57,192], [140,191], [141,185], [160,185], [153,183], [7,180], [149,169], [141,166], [115,160], [53,151]]

print(plot_joint_inference(data))
