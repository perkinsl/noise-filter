#Simulates a learner encountering a corpus of observations of many verbs (data)
#Infers category values (aka, verb transitivity classes) for each verb in data:
    #1: verb is fully transitive (theta = 1)
    #2: verb is fully intransitive (theta = 0)
    #3: verb is mixed (theta sampled from Beta(1,1) uniform distribution
#Infers a single value of epsilon and delta across all verbs in
    #data, a decimal between 0 and 1
#At each iteration, runs 10 steps of the Metropolis-Hastings simulation in
    #MH, and uses the 10th values generated to sample category
    #values using sample_categories
#Data: a list of length n where each item is a 2-element list corresponding
#   to counts of observations for each of n verbs. In each sublist, the first element
#   contains counts of direct objects and the second contains total number of observations
#Iterations: number of iterations to run simulation, must be an integer value
#Returns epsilon, a list of length n of epsilon values, delta, a list of length n of delta values,
	#and verb_categories, an nxv matrix of model values for each of v verbs, for each of n iterations

import random
from MH import *
from sample_categories import sample_categories


def joint_inference(data, iterations):

	#Randomly initialize epsilon and delta
	epsilon = [random.random()]
	delta = [random.random()]
	verb_categories = []
	gammas = {}
	#gammas is a dictionary to memoize all the combination terms in the likelihoods

	for i in range(0, iterations):

		print('iteration', i)

		#Use current epsilon and delta to infer category values
		newcategories = sample_categories(data, epsilon[i], delta[i], gammas)
		print('categories', newcategories)
		verb_categories.append(newcategories)

		#Run Metropolis-Hastings simulation 10 times to infer new epsilon
		#from current delta and category values
		#MH sampling on epsilon
		timelogepsilon = MH(data, verb_categories[i], delta[i], epsilon[i], gammas, 10, Var.EPSILON)
		newepsilon = timelogepsilon[9]
		epsilon.append(newepsilon)

		#Run Metropolis-Hastings simulation 10 times to infer new delta
		#from new epsilon and category values
		#MH sampling on delta
		timelogdelta = MH(data, verb_categories[i], delta[i], newepsilon, gammas, 10, Var.DELTA)
		newdelta = timelogdelta[9]
		delta.append(newdelta)

	return verb_categories, epsilon, delta

#Run joint_inference over 1000 iterations and plot probability distribution over
#categories and epsilon

import numpy as np
import matplotlib.pyplot as plt

def plot_joint_inference(data):

	#verb_categories, epsilon, delta = joint_inference(data, 1000)

	#Use every 10th value from last 500 iterations as samples
	#categorysamples = verb_categories[501::10]
	#epsilonsamples = epsilon[501::10]
	#np.savetxt('epsilon', epsilonsamples)
	#deltasamples = delta[501::10]
	#np.savetxt('delta', deltasamples)
    
    #for the toy data set, we're running joint_inference for 50 iterations
	verb_categories, epsilon, delta = joint_inference(data, 50)
    
	categorysamples = verb_categories[:-1]
	epsilonsamples = epsilon[:-1]
	np.savetxt('epsilon', epsilonsamples)
	deltasamples = delta[:-1]
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

	#Display table containing counts of category assignments per category for each verb
	categories_transposed = list(map(list, zip(*categorysamples)))
	category_table = []

	for i in range(0, len(data)):
		histcounts = np.histogram(categories_transposed[i], bins = [1, 2, 3, 4])
		category_table.append(histcounts[0])

	category_table = np.asarray(category_table)
	np.savetxt('category_table', category_table, fmt='%1i')

	return category_table

#data containing vector of verb counts from CHILDES Treebank
#first list element for each verb in the vector is counts of overt direct objects,
#and second list element is total count for that verb.
#see 'CHILDESTreebank_VerbData' for all 50 verbs in order.
 data = [[308,1568], [777,1318], [11,859], [541,605], [3,605], [155,583], [406,579], [347,550], [350,509], [350,485], [287,477], [13,451], [57,383], [193,375], [221,366], [299,358], [297,356], [274,352], [265,342], [305,337], [299,331], [268,331], [275,312], [4,308], [161,306], [215,299], [21,294], [114,281], [8,275], [198,263], [11,256], [132,255], [11,253], [112,238], [13,228], [49,227], [205,220], [187,214], [8,197], [161,195], [57,192], [140,191], [141,185], [160,185], [153,183], [7,180], [149,169], [141,166], [115,160], [53,151]]

#We can test joint_inference with a smaller toy data set.
#We comment out the data vector, then let
#data = [[19, 20], [9, 10], [1, 20], [2, 40], [10, 20], [3, 10]] and run joint_inference.py.
#Note here we also replaced the code for running Gibbs sampling 1000 iterations and saving every tenth result in the last 500 iterations
#with the code for running Gibbs sampling 50 times and saving all the results from the 50 iterations
#because we are dealing with a smaller data set.
#The output should look something like:
# iteration 0
# categories [1, 1, 2, 2, 3, 2]
# iteration 1
# categories [1, 1, 2, 2, 3, 2]
# iteration 2
# categories [1, 1, 2, 3, 3, 3]
# iteration 3
# categories [1, 3, 2, 2, 3, 3]
# iteration 4
# categories [1, 1, 2, 2, 3, 3]
# iteration 5
# categories [1, 1, 2, 2, 3, 3]
# iteration 6
# categories [3, 1, 2, 2, 3, 2]
# iteration 7
# categories [1, 3, 2, 2, 3, 3]
# iteration 8
# categories [1, 3, 2, 2, 3, 3]
# iteration 9
# categories [1, 1, 3, 2, 3, 3]
# iteration 10
# categories [1, 3, 2, 2, 3, 3]
# iteration 11
# categories [1, 1, 2, 2, 1, 3]
# iteration 12
# categories [3, 1, 2, 2, 1, 3]
# iteration 13
# categories [1, 1, 2, 2, 1, 3]
# iteration 14
# categories [1, 3, 2, 2, 3, 3]
# iteration 15
# categories [1, 1, 2, 2, 3, 3]
# iteration 16
# categories [1, 3, 2, 2, 3, 3]
# iteration 17
# categories [1, 1, 2, 2, 3, 3]
# iteration 18
# categories [1, 1, 2, 2, 3, 3]
# iteration 19
# categories [1, 1, 3, 2, 3, 2]
# iteration 20
# categories [1, 3, 2, 2, 3, 3]
# iteration 21
# categories [1, 3, 3, 2, 3, 3]
# iteration 22
# categories [1, 1, 2, 2, 3, 3]
# iteration 23
# categories [1, 1, 2, 2, 3, 3]
# iteration 24
# categories [1, 1, 2, 2, 3, 3]
# iteration 25
# categories [1, 1, 2, 2, 3, 2]
# iteration 26
# categories [1, 1, 2, 2, 3, 3]
# iteration 27
# categories [1, 3, 3, 2, 3, 3]
# iteration 28
# categories [1, 1, 2, 2, 3, 3]
# iteration 29
# categories [1, 1, 2, 2, 3, 2]
# iteration 30
# categories [1, 1, 2, 2, 3, 3]
# iteration 31
# categories [3, 1, 2, 2, 3, 3]
# iteration 32
# categories [1, 3, 2, 2, 3, 3]
# iteration 33
# categories [1, 1, 2, 2, 3, 3]
# iteration 34
# categories [1, 1, 2, 2, 3, 3]
# iteration 35
# categories [1, 1, 2, 2, 3, 2]
# iteration 36
# categories [1, 1, 3, 2, 3, 2]
# iteration 37
# categories [3, 1, 2, 2, 3, 3]
# iteration 38
# categories [1, 1, 2, 2, 3, 3]
# iteration 39
# categories [1, 1, 2, 2, 3, 3]
# iteration 40
# categories [1, 1, 2, 2, 3, 3]
# iteration 41
# categories [1, 1, 2, 2, 3, 3]
# iteration 42
# categories [1, 1, 2, 2, 3, 3]
# iteration 43
# categories [1, 1, 2, 2, 3, 3]
# iteration 44
# categories [1, 1, 2, 3, 3, 3]
# iteration 45
# categories [1, 3, 2, 2, 3, 2]
# iteration 46
# categories [1, 1, 3, 2, 3, 3]
# iteration 47
# categories [1, 1, 2, 2, 1, 1]
# iteration 48
# categories [1, 1, 2, 2, 3, 2]
# iteration 49
# categories [1, 3, 2, 2, 3, 2]
# [[45  0  4]
#  [38  0 11]
#  [ 0 43  6]
#  [ 0 47  2]
#  [ 4  0 45]
#  [ 1 10 38]]

#The output describes the predicted verb categories of the 6 verbs during each iteration,
#and the last 6 lines indicate the total number of occurrences of the 3 categories for each verb.
#The final predicted verb categories of the input should be interpreted as the one with
#the highest occurrence, so the prediction sbould be [1, 1, 2, 2, 3, 3]
#Additionally, the output also includes two histograms of the distribution over delta and epsilon.
#For this dataset, the distribution over delta is centered at 0.5, 
#and the distribution over epsilon is centered at 0.15

#For this data set, actual delta = 0.5, actual epsilon = 0.1,
#actual models = [1, 1, 2, 2, 3, 3],
#which matches our predicted result

print(plot_joint_inference(data))
