#Simulates a learner encountering a corpus of observations of many verbs (data)
#With known models for verbs that generated that data:
#   1: verb is fully transitive (theta = 1)
#   2: verb is fully intransitive (theta = 0)
#   3: verb is mixed (theta sampled from Beta(1,1) uniform distribution)
#Data: a list of length n where each item is a 2-element list corresponding
#   to counts of observations for each of n verbs. In each sublist, the first element
#   contains counts of direct objects and the second contains total number of observations
#Models: a list of model values (1, 2, or 3) for each verb in the data
#   array, where each element in list corresponds to a row in the data
#   array
#Epsilon: a decimal from 0 to 1
#Returns p, height of function proportional to pdf of posterior probability
#   on delta, at specified value of delta


## NOTE: a lot of this repeats code from sample_models.py
## see comments in that code for details on how this corresponds to math in Perkins, Feldman, & Lidz

import math
import numpy as np
import itertools
from operator import add



def pdf_theta(data, epsilon, delta,theta, gammas):

	if theta <= 0:
		p = float('-inf')
	elif theta >= 1:
		p = float('-inf')
	else:
		verbposteriors = []

		#M1dict = {}
		#M2dict = {}
		M3dict = {}

		## loop through every verb in dataset and calculate p(k|T,epsilon,delta) This is NOT what we want to do for theta
		## following likelihood function in Equation (8) in Perkins, Feldman & Lidz
		for verb in range(0, len(models)):

			verbcount = data[verb]
			k = verbcount[0]
			n = verbcount[1]

			# M1component = []
			# M2component = []
			M3component = []

			## create (n1, k1) tuples containing all combinations of n1 in range (0, n+1) and k1 in range (0, k+1)
			## equivalent to "for n1 in range (0, n+1) for k1 in range (0, k+1)"
			n1 = range(n+1)
			k1 = range(k+1)
			combinations = list(itertools.product(n1, k1))

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

			# def calculate_M1k1(n1, k1):
			# 	if (n1, k1) in M1dict:
			# 		M1k1term = M1dict[(n1, k1)]
			#
			# 	else:
			# 		if k1 <= n1:
			# 			if k1 == n1:
			# 				M1k1term = 0
			# 			else:
			# 				M1k1term = float('-inf')
			# 		else:
			# 			M1k1term = float('-inf')
			# 		M1dict[(n1, k1)] = M1k1term
			#
			# 	return M1k1term

			# def calculate_M2k1(n1, k1):
			# 	if (n1, k1) in M2dict:
			# 		M2k1term = M2dict[(n1, k1)]
			#
			# 	else:
			# 		if k1 <= n1:
			# 			if k1 == 0:
			# 				M2k1term = 0
			# 			else:
			# 				M2k1term = float('-inf')
			#
			# 		else:
			# 			M2k1term = float('-inf')
			#
			# 		M2dict[(n1, k1)] = M2k1term
			#
			# 	return M2k1term

			def calculate_M3k1(n1, k1):
				if (n1, k1) in M3dict:
					M3k1term = M3dict[(n1, k1)]

				else:
					if k1 <= n1:
						#this is the part that needs to be changed
						#M3k1term = np.log(1.0)-np.log(n1+1)
						k2term = gammas[((k1), (n1))]+(k1)*np.log(theta)+((n1)-(k1))*np.log(1-theta)
					else:
						M3k1term = float('-inf')

					M3dict[(n1, k1)] = M3k1term

				return M3k1term

			## group k1s by n1 values in order to efficiently compute inner sums in Equation (8)
			for key, group in itertools.groupby(combinations, lambda x: x[0]):
				ngroup = list(group)
				k2term = list(itertools.starmap(calculate_k2, ngroup))
				#M1k1term = list(itertools.starmap(calculate_M1k1, ngroup))
				#M2k1term = list(itertools.starmap(calculate_M2k1, ngroup))
				M3k1term = list(itertools.starmap(calculate_M3k1, ngroup))

				#M1term = list(map(add, M1k1term, k2term))
				#M2term = list(map(add, M2k1term, k2term))
				M3term = list(map(add, M3k1term, k2term))

				#M1term.sort(reverse=True)
				#M2term.sort(reverse=True)
				M3term.sort(reverse=True)

				# if M1term[0] == float('-inf'):
				# 	M1termsub = M1term
				#
				# else:
				# 	M1termsub = [(i-M1term[0]) for i in M1term]
				#
				# if M2term[0] == float('-inf'):
				# 	M2termsub = M2term
				#
				# else:
				# 	M2termsub = [(i-M2term[0]) for i in M2term]

				if M3term[0] == float('-inf'):
					M3termsub = M3term

				else:
					M3termsub = [(i-M3term[0]) for i in M3term]

				# M1termexp = [math.exp(i) for i in M1termsub]
				# M2termexp = [math.exp(i) for i in M2termsub]
				M3termexp = [math.exp(i) for i in M3termsub]

				# M1logsum = M1term[0] + np.log1p(sum(M1termexp[1:]))
				# M2logsum = M2term[0] + np.log1p(sum(M2termexp[1:]))
				M3logsum = M3term[0] + np.log1p(sum(M3termexp[1:]))

				if (key, n) in gammas:
					noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)

				else:
					gammas[(key, n)] = math.lgamma(n+1)-(math.lgamma(key+1)+math.lgamma(n-key+1))
					noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)

				# M1component.append(M1logsum + noise)
				# M2component.append(M2logsum + noise)
				M3component.append(M3logsum + noise)

			# M1component.sort(reverse=True)
			# M2component.sort(reverse=True)
			M3component.sort(reverse=True)

			# if M1component[0] == float('-inf'):
			# 	M1componentsub = M1component
			#
			# else:
			# 	M1componentsub = [(i-M1component[0]) for i in M1component]
			#
			# if M2component[0] == float('-inf'):
			# 	M2componentsub = M2component
			#
			# else:
			# 	M2componentsub = [(i-M2component[0]) for i in M2component]

			if M3component[0] == float('-inf'):
				M3componentsub = M3component

			else:
				M3componentsub = [(i-M3component[0]) for i in M3component]

			# M1componentexp = [math.exp(i) for i in M1componentsub]
			# M2componentexp = [math.exp(i) for i in M2componentsub]
			M3componentexp = [math.exp(i) for i in M3componentsub]

			# M1likelihood = M1component[0] + np.log1p(sum(M1componentexp[1:]))
			# M2likelihood = M2component[0] + np.log1p(sum(M2componentexp[1:]))
			M3likelihood = M3component[0] + np.log1p(sum(M3componentexp[1:]))

			# if models[verb] == 1:
			# 	verbposteriors.append(M1likelihood)
			#
			# elif models[verb] == 2:
			# 	verbposteriors.append(M2likelihood)
			#
				verbposteriors.append(M3likelihood)
			#
			# else:
			# 	print('Invalid model value')

		## function g(delta) in Equation (13) is equal to product across all verbs of likelihood term, times prior on delta
		## prior is equal to 1 for all values of delta, because delta ~ Beta(1,1),
		## so this reduces to product across all verbs of likelihood term
		## and here, we're returning that value in log space
		verbposteriors

	return verbposteriors
