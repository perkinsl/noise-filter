#Simulates a learner encountering a corpus of observations of many verbs (data)
#With known models for verbs that generated that data:
#   1: verb is fully transitive (theta = 1)
#   2: verb is fully intransitive (theta = 0)
#   3: verb is mixed (theta sampled from Beta(1,1) uniform distribution)
#Data: a list of length n where each item is a 2-element list corresponding
#   to counts of observations for each of n verbs. In each sublist, the first element
#   contains counts of direct objects and the second contains total number of observations
#verb_categories: a list of model values (1, 2, or 3) for each verb in the data
#   array, where each element in list corresponds to a row in the data
#   array
#Delta: a value from 0 to 1
#Epsilon: a value from 0 to 1
#Gammas: dictionary of combination terms from binomial distribution equations,
#    passed on to each iteration of Gibbs sampling in joint_inference.py
#samples from a Gaussian proposal distribution to propose a new value of
#   delta/epsilon/theta, and accepts that proposal depending on the posterior
#   probabilities of delta/epsilon/theta and the proposed new delta/epsilon/theta given in the
#   pdf_delta_epsilon/pdf_theta function
#Returns a two-element list of the accepted value and its acceptance probability


import math
import random
from pdf_delta_epsilon import pdf
from pdf_theta import pdf_theta_one_verb

## Acceptance function that takes a current variable, a new proposed variable
## and the probabilities of each in the function proportional to the posterior pdf
## and decides whether to accept the new proposed variable, or keep the old one
def accept(var, var_prime, p, p_prime):
	#print("var: ", var, "p_var: ", math.exp(p))
	#print("proposed new value:", var_prime)
	#print("p_var_prime", math.exp(p_prime))
	#Reject impossible proposals
	if p_prime == float('-inf'):
		#print("REJECTED bc p 0")
		return (var, p)

	#Accept possible proposal var_prime with acceptance probability A (in log space)
	else:
		A = min(0, p_prime-p)

		if A == 0:
			#print("var_prime, p_prime", var_prime, p_prime)
			#print("ACCEPTED NEW VALUE")
			return (var_prime, p_prime)

		else:
			x = random.random()
			if x < math.exp(p_prime-p):
				#print("var_prime, p_prime", var_prime, p_prime)
				#print("ACCEPTED NEW VALUE")
				return (var_prime, p_prime)
			else:
				#print("var_prime, p_prime", var_prime, p_prime)
				#print("REJECTED")
				return (var, p)

#Need to pass data, verb_categories, delta, epsilon, gammas, iteration varaibles to call pdf functions
def propose_and_accept(data, verb_categories, delta, epsilon, gammas, var, p_var, flag):


	#Sample a new value of var from a proposal distribution Q, a Gaussian
	#with mu = var and sigma = 0.25
	var_prime  = random.gauss(var, 0.25)


	#Call the corresponding pdf function according to the variable flag
	if flag == 0:
		#Use pdf to calculate logs of height of var_prime on curve proportional to pdf over var
		p_var_prime = pdf(data, verb_categories, var_prime, epsilon, gammas)

	elif flag == 1:
		p_var_prime = pdf(data, verb_categories, delta, var_prime, gammas)

	else:
		#Here we take in one verb for the data variable 
		p_var_prime = pdf_theta_one_verb(data, delta, epsilon, var_prime, gammas)
	#returns both var and p_var to be updated outside this function
	return accept(var, var_prime, p_var, p_var_prime)
