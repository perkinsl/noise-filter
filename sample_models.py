#Simulates a learner encountering a corpus of observations of many verbs (data)
#With known epsilon and delta (a value between 0 and 1)
#Data: a list of length n where each item is a 2-element list corresponding 
#   to counts of observations for each of n verbs. In each sublist, the first element 
#   contains counts of direct objects and the second contains total number of observations
#Gammas: dictionary of combination terms from binomial distribution equations, 
#    passed on to each iteration of Gibbs sampling in joint_inference.py
#Infers posterior probabilities on models (aka, verb transitivity classes) for each verb in data:
#    M1: verb is fully transitive (theta = 1)
#    M2: verb is fully intransitive (theta = 0)
#    M3: verb is mixed (theta sampled from Beta(1,1) uniform distribution)
#Samples a model value for each verb by flipping a biased coin weighted by
#   those posterior probabilities over models
#Returns a vector of model values (1, 2, or 3) for each verb in the data
#   matrix, where each element in vector corresponds to a row in the data
#   matrix

#Updates 08/03/2021: replaced repeated code with function likelihoods

import math
import random
import numpy as np
from pdf_delta_epsilon import likelihoods

def sample_models(data, epsilon, delta, gammas):
	models = []

	## memoizing specific n1, k1 combinations for Equation (10) in Perkins, Feldman & Lidz
	## because these will always produce the same result, regardless of the verb identity
	M1dict = {}
	M2dict = {}
	M3dict = {}
	
	## loop through every verb in dataset and calculate posterior on transitivity models (T)
	## following Equation (7) in Perkins, Feldman, & Lidz
	for i in range(0, len(data)):
		
		verbcount = data[i]
		
		## prior P(T) from Equation (7) is flat: 1/3 for each value of T
		M1prior = 1.0/3.0
		M2prior = 1.0/3.0
		M3prior = 1.0/3.0
		
		verbLikelihoods = likelihoods(verbcount, delta, epsilon, gammas, M1dict, M2dict, M3dict)
		
		M1likelihood = verbLikelihoods[0]
		M2likelihood = verbLikelihoods[1]
		M3likelihood = verbLikelihoods[2]
        
		## these are the log numerators for all three verb categories in Equation (7)
		demM1 = M1likelihood + np.log(M1prior)
		demM2 = M2likelihood + np.log(M2prior)
		demM3 = M3likelihood + np.log(M3prior)
		
		## calculating denominator for Equation (7)
		dems = [demM1, demM2, demM3]
		demsexp = [math.exp(i) for i in dems] ## list comprehension: [fun(i) for i in list]
		denominator = np.log(sum(demsexp))
		
		## final result of Equation (7), in log space
		M1 = (M1likelihood + np.log(M1prior)) - denominator
		M2 = (M2likelihood + np.log(M2prior)) - denominator
		#no need to calculate M3 since the prediction will automatically fall into M3 if it doesn't fit into M1 and M2
		#M3 = (M3likelihood + np.log(M3prior)) - denominator
		
		#flips a biased coin weighted by posteriors on models to sample a model (ie, category) for each verb
		x = random.random()
		if x <= math.exp(M1):
			models.append(1)
			
		elif x <= (math.exp(M1) + math.exp(M2)):
			models.append(2)
		
		else:
			models.append(3)
			
	return models
