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
#Epsilon: a value from 0 to 1
#Iterations: number of iterations for Metropolis-Hastings simulation, must
#   be an integer value
#Conducts Metropolis-Hastings simulation over specified number of
#   iterations: initializes a random value of delta, samples
#   from a Gaussian proposal distribution to propose a new value of
#   delta, and accepts that proposal depending on the posterior
#   probabilities of delta and the proposed new delta given in the
#   pdf_delta function
#Returns vector of delta values after running specified number of iterations

import math
import random
from pdf_delta import pdf_delta

def MH_delta(data, models, epsilon, gammas, iterations):

	#Initialize a random value of delta
	delta = random.random()
	
	#Use pdf_delta.m to calculate logs of height of delta on curve proportional to pdf over delta
	p_delta = pdf_delta(data, models, epsilon, delta, gammas)
	
	timelog = [delta]
	
	for i in range(1, iterations):
	
		#Sample a new value of delta from a proposal distribution Q, a Gaussian
		#with mu = delta and sigma = 0.25
		deltaprime = random.gauss(delta, 0.25)
		#print('deltaprime', deltaprime)
		
		#Use pdf_delta.m to calculate logs of height of deltaprime on curve proportional to pdf over deltaprime
		p_deltaprime = pdf_delta(data, models, epsilon, deltaprime, gammas)
		#print('p_deltaprime', p_deltaprime)
		
		if p_deltaprime == float('-inf'):
			delta = delta
			p_delta = p_delta
		
		else:
			#Accept proposal deltaprime with acceptance probability A (in log space)
			A = min(0, p_deltaprime-p_delta)
		
			if A == 0:
				delta = deltaprime
				p_delta = p_deltaprime
			
			else:
				x = random.random()
				if x < math.exp(p_deltaprime-p_delta):
					delta = deltaprime
					p_delta = p_deltaprime
				
		timelog.append(delta)
		
	return timelog
