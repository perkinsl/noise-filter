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

#Updates 07/29/2021:
#1. merged MH_delta and MH_epsilon into one file (the two files are identical except for the input arguments 
#       and when calculating the log pdf with constant delta/epsilon and updated epsilon/delta);
#2. kept both delta and epsilon in the input argument and added boolean variable isDelta to input argument 
#       to determine whether the current MH sampling is being done on delta or epsilon so that we can call function pdf with appropriate variables

import math
import random
from pdf import pdf

def MH(data, models, delta, epsilon, gammas, iterations, isDelta):
    
	#Initialize a random value of epsilon
	MHvar = random.random()
	
	#Use pdf.m to calculate logs of height of epsilon on curve proportional to pdf over epsilon
	p_MHvar = pdf(data, models, delta, epsilon, gammas)
	
	timelog = [MHvar]
	
	for i in range(1, iterations):
		
		#Sample a new value of epsilon from a proposal distribution Q, a Gaussian
		#with mu = epsilon and sigma = 0.25
		MHvar_prime = random.gauss(MHvar, 0.25)
		#print('epsilonprime', epsilonprime)
		
        	#Determine whether the variable being sampled with MH sampling is delta or epsilon
		if isDelta:
			#Use pdf.m to calculate logs of height of epsilonprime on curve proportional to pdf over epsilon
			p_MHvar_prime = pdf(data, models, MHvar_prime, epsilon, gammas)
			#print('p_epsilonprime', p_epsilonprime)
		
		else:
			p_MHvar_prime = pdf(data, models, delta, MHvar_prime, gammas)
        
		if p_MHvar_prime == float('-inf'):
			MHvar_prime = MHvar_prime
			p_MHvar = p_MHvar
		
		else:
			#Accept proposal epsilonprime with acceptance probability A (in log space)
			A = min(0, p_MHvar_prime-p_MHvar)
		
			if A == 0:
				MHvar = MHvar_prime
				p_MHvar = p_MHvar_prime

			else:
				x = random.random()
				if x < math.exp(p_MHvar_prime-p_MHvar):
					MHvar = MHvar_prime
					p_MHvar = p_MHvar_prime
				
		timelog.append(MHvar)
		
	return timelog
