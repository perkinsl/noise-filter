# #Simulates a learner encountering a corpus of observations of many verbs (data)
# #With known models for verbs that generated that data:
# #   1: verb is fully transitive (theta = 1)
# #   2: verb is fully intransitive (theta = 0)
# #   3: verb is mixed (theta sampled from Beta(1,1) uniform distribution)
# This file assumes only one category, alternating verbs.
# #Data: a list of length n where each item is a 2-element list corresponding
# #   to counts of observations for each of n verbs. In each sublist, the first element
# #   contains counts of direct objects and the second contains total number of observations
# #Epsilon: a decimal from 0 to 1
# #Returns p, height of function proportional to pdf of posterior probability
# #   on delta, at specified value of delta
#
#
# ## NOTE: a lot of this repeats code from sample_models.py
# ## see comments in that code for details on how this corresponds to math in Perkins, Feldman, & Lidz

import math
import numpy as np
import itertools
from operator import add



def pdf_theta(data, delta, epsilon, thetas, gammas):

	verbposteriors = []
		## loop through every verb in dataset and calculate p(k|T,epsilon,delta)
		## following likelihood function in Equation (8) in Perkins, Feldman & Lidz
	for verb in range(0, len(data)):
		verbposteriors.append(pdf_theta_one_verb(data[verb], delta, epsilon, thetas[verb], gammas))

	# return a vector of the probabilities of each theta value for each verb, in log space
	p = verbposteriors
	print(p)
	return p

def pdf_theta_one_verb(verb, delta, epsilon, theta, gammas):
	if theta <= 0:
		M3likelihood = float('-inf')
	elif theta >= 1:
		M3likelihood = float('-inf')

	else:

		verbcount = verb
		k = verbcount[0]
		n = verbcount[1]



		M3component = []

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
		def calculate_M3k1(n1, k1):
			if k1 <= n1:
				if ((k1), (n1)) in gammas:
					M3k1term = gammas[((k1), (n1))] + (k1)*np.log(theta) + ((n1)-(k1))*np.log(1-theta)
				else:
					gammas[((k1), (n1))] = math.lgamma(n1+1)-(math.lgamma(k1+1)+math.lgamma((n1)-(k1)+1))
					M3k1term = gammas[((k1), (n1))] + (k1)*np.log(theta)+ (((n1)-(k1))*np.log(1-theta))
			else:
				M3k1term = float('-inf')

			return M3k1term

		## group k1s by n1 values in order to efficiently compute inner sums in Equation (8)
		for key, group in itertools.groupby(combinations, lambda x: x[0]):
			ngroup = list(group)
			k2term = list(itertools.starmap(calculate_k2, ngroup))
			M3k1term = list(itertools.starmap(calculate_M3k1, ngroup))

			M3term = list(map(add, M3k1term, k2term))


			M3term.sort(reverse=True)



			if M3term[0] == float('-inf'):
				M3termsub = M3term

			else:
				M3termsub = [(i-M3term[0]) for i in M3term]


			M3termexp = [math.exp(i) for i in M3termsub]

			M3logsum = M3term[0] + np.log1p(sum(M3termexp[1:]))
			#Binomial(n, 1-epsilon)
			if (key, n) in gammas:
				noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)

			else:
				gammas[(key, n)] = math.lgamma(n+1)-(math.lgamma(key+1)+math.lgamma(n-key+1))
				noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)


			M3component.append(M3logsum + noise)

		M3component.sort(reverse=True)


		if M3component[0] == float('-inf'):
			M3componentsub = M3component

		else:
			M3componentsub = [(i-M3component[0]) for i in M3component]

		M3componentexp = [math.exp(i) for i in M3componentsub]

		M3likelihood = M3component[0] + np.log1p(sum(M3componentexp[1:]))
	return M3likelihood

#data = [[19, 20], [9, 10] ,[1, 20], [2, 40], [10, 20], [3, 10]]
data = [[19, 20], [10, 10], [1, 20], [1, 40], [10, 20], [4, 10]]
data3 = [[16, 20], [8, 10], [4, 20], [8, 40], [10, 20], [3, 10]]
thetas = [0.95, 1.0, 0.05, 0.025, 0.5, 0.4]
thetas3 = [0.8, 0.8, 0.2, 0.2, 0.5, 0.3]
data4 = [[14, 20], [7, 10], [6, 20], [12, 40], [10, 20], [4, 10]]
thetas4 = [0.7, 0.7,0.3, 0.3, 0.5, 0.4]
data5 = [[11, 20],  [5, 10], [9, 20], [19, 40], [10, 20], [5, 10]]
thetas5 = [0.55, 0.5, 0.45, 0.475, 0.5, 0.5]
data6 = [[19, 20], [38, 40], [14, 15], [10, 10], [29, 30], [23, 25]]
thetas6 = [0.95, 0.95, 0.933, 1.0, 0.966, 0.92]
data7 =[[1, 20], [2, 40], [0, 15], [1, 10], [2, 30], [1, 25]]
thetas7 = [0.05, 0.05, 0, 0.1, 0.0667, 0.04]
data8 = [[7, 20], [6, 40], [7, 15], [4, 10], [7, 30], [19, 25]]
thetas8 = [0.35, 0.15, 0.4667, 0.4, 0.233, 0.76]

# pdf_theta_one_verb(data1, 0.01, 0.01, 0.95, {})
pdf_theta(data8, 0.01, 0.01, thetas8, {})