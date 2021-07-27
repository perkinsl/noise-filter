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
#Delta: a decimal from 0 to 1
#Iterations: number of iterations for Metropolis-Hastings simulation, must
#   be an integer value
#Conducts Metropolis-Hastings simulation over specified number of
#   iterations: initializes a random value of epsilon, samples
#   from a Gaussian proposal distribution to propose a new value of
#   epsilon, and accepts that proposal depending on the posterior
#   probabilities of epsilon and the proposed new epsilon given in the
#   pdf_epsilon function
#Returns vector of epsilon values after running specified number of iterations

import math
import random
from pdf_epsilon import pdf_epsilon

def MH_epsilon(data, models, delta, gammas, iterations):

	#Initialize a random value of epsilon
	epsilon = random.random()
	
	#Use pdf_epsilon.m to calculate logs of height of epsilon on curve proportional to pdf over epsilon
	p_epsilon = pdf_epsilon(data, models, delta, epsilon, gammas)
	
	timelog = [epsilon]
	
	for i in range(1, iterations):
		
		#Sample a new value of epsilon from a proposal distribution Q, a Gaussian
		#with mu = epsilon and sigma = 0.25
		epsilonprime = random.gauss(epsilon, 0.25)
		#print('epsilonprime', epsilonprime)
		
		#Use pdf_epsilon.m to calculate logs of height of epsilonprime on curve proportional to pdf over epsilon
		p_epsilonprime = pdf_epsilon(data, models, delta, epsilonprime, gammas)
		#print('p_epsilonprime', p_epsilonprime)
		
		if p_epsilonprime == float('-inf'):
			epsilon = epsilon
			p_epsilon = p_epsilon
		
		else:
			#Accept proposal epsilonprime with acceptance probability A (in log space)
			A = min(0, p_epsilonprime-p_epsilon)
		
			if A == 0:
				epsilon = epsilonprime
				p_epsilon = p_epsilonprime

			else:
				x = random.random()
				if x < math.exp(p_epsilonprime-p_epsilon):
					epsilon = epsilonprime
					p_epsilon = p_epsilonprime
				
		timelog.append(epsilon)
		
	return timelog
