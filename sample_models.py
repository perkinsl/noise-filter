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

import math
import random
import numpy as np
import itertools
from operator import add

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
		
		verbcounts = data[i]
		
		## prior P(T) from Equation (7) is flat: 1/3 for each value of T
		M1prior = 1.0/3.0
		M2prior = 1.0/3.0
		M3prior = 1.0/3.0
		
		k = verbcounts[0]
		n = verbcounts[1]
		
		M1component = []
		M2component = []
		M3component = []
		
		## likelihood p(k|T, epsilon, delta): implementing Equation (8) in Perkins, Feldman & Lidz
		## create (n1, k1) tuples containing all combinations of n1 in range (0, n+1) and k1 in range (0, k+1)
		## equivalent to "for n1 in range (0, n+1) for k1 in range (0, k+1)"
		n1 = range(n+1)
		k1 = range(k+1)
		combinations = list(itertools.product(n1, k1)) ## returns cartesian product of n1 x k1
		
		## implementing Equation (9) in Perkins, Feldman & Lidz: p(k0|n0, delta), in log space
		def calculate_k2(n1, k1): ## using variable names 'k2' and 'n2' where 'k0' and 'n0' are used in the paper
			if (k-k1)<=(n-n1):
				if ((k-k1), (n-n1)) in gammas:
					k2term = gammas[((k-k1), (n-n1))]+(k-k1)*np.log(delta)+((n-n1)-(k-k1))*np.log(1-delta)
				else:
					gammas[((k-k1), (n-n1))] = math.lgamma(n-n1+1)-(math.lgamma(k-k1+1)+math.lgamma((n-n1)-(k-k1)+1))
					k2term = gammas[((k-k1), (n-n1))]+(k-k1)*np.log(delta)+((n-n1)-(k-k1))*np.log(1-delta)
			
			else:
				k2term = float('-inf') ## log of zero
			
			return k2term
		
		## implementing Equation (10) in Perkins, Feldman & Lidz: p(k1|n1, T), in log space
		## M1k1 is result for transitives (T=1)
		def calculate_M1k1(n1, k1):
			if (n1, k1) in M1dict:
				M1k1term = M1dict[(n1, k1)]
				
			else:
				if k1 <= n1:	
					if k1 == n1:
						M1k1term = 0 ## log of 1: transitive category has pr. 1 if k1 = n1
					else:
						M1k1term = float('-inf') ## log of zero: transitive category has pr. 0 for all other k1, n1 combinations
				else:
					M1k1term = float('-inf')
				M1dict[(n1, k1)] = M1k1term
				
			return M1k1term
		
		## M2k1 is result for intransitives (T=2)
		def calculate_M2k1(n1, k1):
			if (n1, k1) in M2dict:
				M2k1term = M2dict[(n1, k1)]

			else:				
				if k1 <= n1:
					if k1 == 0:
						M2k1term = 0 ## log of 1: intransitive category has pr. 1 if k1 = 0
					else:
						M2k1term = float('-inf') ## log of zero: intransitive category has pr. 0 for all other k1, n1 combinations
			
				else:
					M2k1term = float('-inf')
			
				M2dict[(n1, k1)] = M2k1term
				
			return M2k1term
		
		## M3k1 is result for alternators (T=3)	
		def calculate_M3k1(n1, k1):
			if (n1, k1) in M3dict:
				M3k1term = M3dict[(n1, k1)]
				
			else:
				if k1 <= n1:
					M3k1term = np.log(1.0)-np.log(n1+1) ## result of integrating over all values of theta: 1/(n1+1)
				else:
					M3k1term = float('-inf')
					
				M3dict[(n1, k1)] = M3k1term
				
			return M3k1term
		
		## group k1s by n1 values in order to efficiently compute inner sums in Equation (8)
		for key, group in itertools.groupby(combinations, lambda x: x[0]):
			ngroup = list(group)

			## itertools.starmap() applies given function using all elements from the tuple as arguments
			## e.g., it applies calculate_k2 to the (n1, k1) tuples in the given list of tuples
			k2term = list(itertools.starmap(calculate_k2, ngroup))
			M1k1term = list(itertools.starmap(calculate_M1k1, ngroup))
			M2k1term = list(itertools.starmap(calculate_M2k1, ngroup))
			M3k1term = list(itertools.starmap(calculate_M3k1, ngroup))
			
			## computing inner sum for this group of k1s
			## start by multiplying terms in Equations (9) and (10) in log space for all values of k1
			M1term = list(map(add, M1k1term, k2term))
			M2term = list(map(add, M2k1term, k2term))
			M3term = list(map(add, M3k1term, k2term))
			

			## trick for computing the log of a summation without stack overflow:
			## you can subtract the largest log value from all other values without exponentiating it
			## log(sum of a_i from i=0 to N) = 
			##		= log(a_0) + log(1 + (sum of (a_i)/(a_0) from i=1 to N))
			##		= log(a_0) + log(1 + (sum of exp(log(a_i) - log(a_0)) from i=1 to N))
			## for a_0 > a_1 > ... > a_N

			## sort lists from large to small
			M1term.sort(reverse=True)
			M2term.sort(reverse=True)
			M3term.sort(reverse=True)
			
			## if largest log probability in list is -inf, result of subtraction for rest of list is also -inf
			if M1term[0] == float('-inf'):
				M1termsub = M1term
			
			## otherwise, perform subtraction for rest of list
			else:
				M1termsub = [(i-M1term[0]) for i in M1term]
			
			if M2term[0] == float('-inf'):
				M2termsub = M2term

			else:
				M2termsub = [(i-M2term[0]) for i in M2term]
			
			if M3term[0] == float('-inf'):
				M3termsub = M3term
			
			else:
				M3termsub = [(i-M3term[0]) for i in M3term]

			## exponentiate subtraction result
			M1termexp = [math.exp(i) for i in M1termsub]
			M2termexp = [math.exp(i) for i in M2termsub]
			M3termexp = [math.exp(i) for i in M3termsub]

			## add to 1, re-log, and add to first log probability in list
			## np.log1p() calculates log(1 + x) for each element x of input array
			M1logsum = M1term[0] + np.log1p(sum(M1termexp[1:]))
			M2logsum = M2term[0] + np.log1p(sum(M2termexp[1:]))
			M3logsum = M3term[0] + np.log1p(sum(M3termexp[1:]))
			
			## inner sum of Equation (8) is now finished! 
			## calculate noise term: p(n1|epsilon), following Equation (11)
			## 'key' is the name for the current value of n1 for this group of k1s
			if (key, n) in gammas:
				noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)
				
			else:
				gammas[(key, n)] = math.lgamma(n+1)-(math.lgamma(key+1)+math.lgamma(n-key+1))
				noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)
			
			## add noise term to result of inner sum
			M1component.append(M1logsum + noise)
			M2component.append(M2logsum + noise)
			M3component.append(M3logsum + noise)
		
		## compute outer sum of Equation (8) using the same trick that we used for inner sum
		M1component.sort(reverse=True)
		M2component.sort(reverse=True)
		M3component.sort(reverse=True)
		
		if M1component[0] == float('-inf'):
			M1componentsub = M1component
		
		else:
			M1componentsub = [(i-M1component[0]) for i in M1component]
		
		if M2component[0] == float('-inf'):
			M2componentsub = M2component
		
		else:
			M2componentsub = [(i-M2component[0]) for i in M2component]
		
		if M3component[0] == float('-inf'):
			M3componentsub = M3component

		else:
			M3componentsub = [(i-M3component[0]) for i in M3component]

		M1componentexp = [math.exp(i) for i in M1componentsub]
		M2componentexp = [math.exp(i) for i in M2componentsub]
		M3componentexp = [math.exp(i) for i in M3componentsub]
		
		## final result of Equation (8) (likelihoods) for all three verb categories
		M1likelihood = M1component[0] + np.log1p(sum(M1componentexp[1:]))
		M2likelihood = M2component[0] + np.log1p(sum(M2componentexp[1:]))
		M3likelihood = M3component[0] + np.log1p(sum(M3componentexp[1:]))

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
		M3 = (M3likelihood + np.log(M3prior)) - denominator
		
		#flips a biased coin weighted by posteriors on models to sample a model (ie, category) for each verb
		x = random.random()
		if x <= math.exp(M1):
			models.append(1)
			
		elif x <= (math.exp(M1) + math.exp(M2)):
			models.append(2)
		
		else:
			models.append(3)
			
	return models
