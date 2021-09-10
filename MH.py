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

#Updates 08/12/2021: replaced code with accept function in MH_theta
#Updates 08/24/2021: update p_MHvar after calling accept function instead of doing computation every time at the beginning of for loop

import math
import random
from pdf_delta_epsilon import pdf
from pdf_theta1 import pdf_theta_one_verb
from propose_and_accept import accept, propose_and_accept

def MH(data, verb_categories, delta, epsilon, gammas, iterations, flag):
	#flag determines whether we are sampling delta, epsilon, or theta. delta == 0, epsilon == 1, theta == 2

	if flag < 2:
		#Initialize a random value of epsilon/delta if sampling for one of those
		MHvar = random.random()
		sampled_results = [MHvar]
	else:
		#initialize a random value of theta for each verb if sampling for theta
		thetas = [random.random() for i in range(0, len(data))]
		sampled_results = [thetas]


		#Determine whether the variable being sampled with MH sampling is delta, epsilon, or theta
		#Use pdf to calculate logs of height of relevant variable on curve proportional to pdf over epsilon
	if flag == 0:
		p_MHvar = pdf(data, verb_categories, MHvar, epsilon, gammas)
		print("p_MHvar=  ", p_MHvar)
	elif flag == 1:
		p_MHvar = pdf(data, verb_categories, delta, MHvar, gammas)
		print("p_MHvar=  ", p_MHvar)
	else:
		p_thetas = [pdf_theta_one_verb(data[j], delta, epsilon, thetas[j], gammas) for j in range(0, len(thetas))]
		print("p_thetas=  ", p_thetas)
	for i in range(1, iterations):
		print("-------------\niteration", i)


		if flag < 2:
			#if we are using MH sampling for epsilon or delta, just call propose_and_accept one time per iteration
			result = propose_and_accept(data, verb_categories, delta, epsilon, gammas, 0,  MHvar, p_MHvar, flag)
			print("result =", result, "\n\n")
			MHvar = result[0] #since propose_and_accept returns a tuple, set first element in tuple as MHvar
			p_MHvar = result[1] #set second element in tuple as p_MHvar
			#add the result to the sampled_results list
			sampled_results.append(MHvar)
			# print("MHvar = ", MHvar)
			# print("p_MHvar =", p_MHvar)
		#this function returns a new MHvar
		else:
			#for theta, call propose_and_accept on each theta value (ie, for each verb)
			result = [propose_and_accept(data, verb_categories, delta, epsilon, gammas, j, thetas[j], p_thetas[j], flag) for j in range(0, len(thetas))]
			# print("result =", result, "\n\n")
			thetas = [i[0] for i in result] #propose_and_accept returns a list of tuples, set first element of each tuple as thetas list
			print("thetas= ", thetas, )
			p_thetas = [i[1] for i in result] #set second element of each tuple as p_thetas list
			sampled_results.append(thetas) #add the accepted thetas to results list

	return sampled_results
data = [[19, 20], [10, 10]]
epsilon = [random.random()]
delta = [random.random()]
verb_categories = []
gammas = {}
MH(data, verb_categories, delta[0], epsilon[0], gammas, 10, 2)
