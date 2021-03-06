#calculates f(x) for specific value x of theta, where f is a function returning a value proportional to the posterior probability on theta. Note that this file works on one verb at a time, and the MH.py script with flag 2 calls it on each verb in the data structure.
#With known models for verbs that generated that data:
#   1: verb is fully transitive (theta = 1)
#   2: verb is fully intransitive (theta = 0)
#   3: verb is mixed (theta sampled from Beta(1,1) uniform distribution)
#Verbcount: the sublist with the counts of the data from the verb being sampled
#This file assumes only one category, alternating verbs.
#Data: a list of length n where each item is a 2-element list corresponding
#   to counts of observations for each of n verbs. In each sublist, the first element
#   contains counts of direct objects and the second contains total number of observations
#Epsilon: a decimal from 0 to 1
#Delta: a decimal from 0 to 1
#Gammas: A dictionary to memoize all the combination terms in the likelihoods
#Returns p, height of function proportional to pdf of posterior probability
#   on theta, at specified value of theta
#Samples for theta on one verb only

import math
import numpy as np
import itertools
from operator import add




def pdf_theta_one_verb(verbcount, delta, epsilon, theta, gammas):
	if theta <= 0:
		T3likelihood = float('-inf')
	elif theta >= 1:
		T3likelihood = float('-inf')

	else:


		k = verbcount[0]
		n = verbcount[1]



		T3component = []
		## Following code is very similar to code in likelihoods.py. For more extensive explanations of the calculatations, see that script. 
		## create (n1, k1) tuples containing all combinations of n1 in range (0, n+1) and k1 in range (0, k+1)
		## equivalent to "for n1 in range (0, n+1) for k1 in range (0, k+1)"

		n1 = range(n+1)
		k1 = range(k+1)
		combinations = list(itertools.product(n1, k1))
		## implementing Equation (9) in Perkins, Feldman & Lidz: p(k0|n0, delta), in log space
		def calculate_k2(n1, k1):
			if (k-k1)<=(n-n1):
				if ((k-k1), (n-n1)) in gammas:
					k2term = gammas[((k-k1), (n-n1))]+(k-k1)*np.log(delta)+((n-n1)-(k-k1))*np.log(1-delta)
				else:
					gammas[((k-k1), (n-n1))] = math.lgamma(n-n1+1)-(math.lgamma(k-k1+1)+math.lgamma((n-n1)-(k-k1)+1))
					k2term = gammas[((k-k1), (n-n1))]+(k-k1)*np.log(delta)+((n-n1)-(k-k1))*np.log(1-delta)

			else:
				k2term = float('-inf')

			return k2term

		#implementing the calculation for p(k1 | n1, theta) (Binomial(n, theta)), in log space
		def calculate_T3k1(n1, k1):
			if k1 <= n1:
				if ((k1), (n1)) in gammas:
					T3k1term = gammas[((k1), (n1))] + (k1)*np.log(theta) + ((n1)-(k1))*np.log(1-theta)
				else:
					gammas[((k1), (n1))] = math.lgamma(n1+1)-(math.lgamma(k1+1)+math.lgamma((n1)-(k1)+1))
					T3k1term = gammas[((k1), (n1))] + (k1)*np.log(theta)+ (((n1)-(k1))*np.log(1-theta))
			else:
				T3k1term = float('-inf')

			return T3k1term

		## group k1s by n1 values in order to efficiently compute inner sums in Equation (8)
		for key, group in itertools.groupby(combinations, lambda x: x[0]):
			ngroup = list(group)
			k2term = list(itertools.starmap(calculate_k2, ngroup))
			T3k1term = list(itertools.starmap(calculate_T3k1, ngroup))

			T3term = list(map(add, T3k1term, k2term))


			T3term.sort(reverse=True)



			if T3term[0] == float('-inf'):
				T3termsub = T3term

			else:
				T3termsub = [(i-T3term[0]) for i in T3term]


			T3termexp = [math.exp(i) for i in T3termsub]

			T3logsum = T3term[0] + np.log1p(sum(T3termexp[1:]))
			#Binomial(n, 1-epsilon)
			if (key, n) in gammas:
				noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)

			else:
				gammas[(key, n)] = math.lgamma(n+1)-(math.lgamma(key+1)+math.lgamma(n-key+1))
				noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)


			T3component.append(T3logsum + noise)

		T3component.sort(reverse=True)


		if T3component[0] == float('-inf'):
			T3componentsub = T3component

		else:
			T3componentsub = [(i-T3component[0]) for i in T3component]

		T3componentexp = [math.exp(i) for i in T3componentsub]

		T3likelihood = T3component[0] + np.log1p(sum(T3componentexp[1:]))
	return T3likelihood


