#Conducts Metropolis-Hastings simulation to sample distribution over theta
#	(probability of direct object) for each verb in learner's data 
#Initializes a random value of theta, samples
#   from a Gaussian proposal distribution to propose a new value of
#   theta for each verb, and accepts that proposal depending on the posterior
#   probabilities of theta and the proposed new theta given in the
#   pdf_theta function
#Data: a list of length n where each item is a 2-element list corresponding 
#   to counts of observations for each of n verbs. In each sublist, the first element 
#   contains counts of direct objects and the second contains total number of observations
#Epsilon: a decimal from 0 to 1
#Delta: a decimal from 0 to 1
#Gammas: memoization variable (dictionary) passed to pdf_theta
#Iterations: number of iterations for Metropolis-Hastings simulation, must
#   be an integer value
#Returns list of lists of theta values for each verb after running specified number of iterations

#Updates 08/24/2021: make accept function return a tuple of the updated variable value and its corresponding acceptance probability

import math
import random
from pdf_theta import pdf_theta

## Acceptance function that takes a current variable, a new proposed variable
## and the probabilities of each in the function proportional to the posterior pdf
## and decides whether to accept the new proposed variable, or keep the old one 
def accept(var, var_prime, p, p_prime):

	#Reject impossible proposals
	if p_prime == float('-inf'):
		return (var, p)

	#Accept possible proposal var_prime with acceptance probability A (in log space)
	else:
		A = min(0, p_prime-p)
		
		if A == 0:
			return (var_prime, p_prime)

		else:
			x = random.random()
			if x < math.exp(p_prime-p):
				return (var_prime, p_prime)
			else:
				return (var, p)
		
def MH_theta(data, delta, epsilon, gammas, iterations):

	#Initialize random values of thetas for each verb
	thetas = [random.random() for i in range(0, len(data))]
	timelog = [thetas]
    
    	#Use pdf_theta to calculate logs of height of theta on curve proportional to pdf over epsilon
	p_thetas = pdf_theta(data, delta, epsilon, thetas, gammas)
	
	for i in range(1, iterations):
		print('iteration', i)

		#Sample a new value of theta for each verb from a proposal distribution Q, a Gaussian
		#with mu = theta and sigma = 0.25
		thetaprimes = [random.gauss(thetas[j], 0.25) for j in range(0, len(thetas))]
		#print('theta_primes', thetaprimes)
		
		#Use pdf_theta to calculate logs of height of each thetaprime on curve proportional to pdf over theta
		p_thetaprimes = pdf_theta(data, delta, epsilon, thetaprimes, gammas)
		#print('p_thetaprimes', p_thetaprimes)

		#Decide whether to accept each new value of theta using acceptance function
        	#a list of 2-element tuples (theta value and corresponding probability)
		thetas_and_probs = [accept(thetas[j], thetaprimes[j], p_thetas[j], p_thetaprimes[j]) for j in range(0, len(thetas))]
		thetas = [theta_and_prob[0] for theta_and_prob in thetas_and_probs]
		p_thetas = [theta_and_prob[1] for theta_and_prob in thetas_and_probs]
        
		print(thetas)
		
		timelog.append(thetas)
		
	return timelog
