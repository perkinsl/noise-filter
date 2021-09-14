
#Updates 09/13/2021: removed verbNumber variable in input arguments of function propose_and_accept

import math
import random
from pdf_delta_epsilon import pdf
from pdf_theta1 import pdf_theta_one_verb

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
