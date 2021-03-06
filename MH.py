#Simulates a learner encountering a corpus of observations of many verbs (data)
#With known categories for verbs that generated that data:
#   1: verb is fully transitive (theta = 1)
#   2: verb is fully intransitive (theta = 0)
#   3: verb is mixed (theta sampled from Beta(1,1) uniform distribution)
#Data: a list of length n where each item is a 2-element list corresponding
#   to counts of observations for each of n verbs. In each sublist, the first element
#   contains counts of direct objects and the second contains total number of observations
#verb_categories: a list of category values (1, 2, or 3) for each verb in the data
#   array, where each element in list corresponds to a row in the data
#   array
#Delta: a value from 0 to 1
#Epsilon: a value from 0 to 1
#Gammas: dictionary of combination terms from binomial distribution equations,
#    passed on to each iteration of Gibbs sampling in joint_inference.py
#Iterations: number of iterations for Metropolis-Hastings simulation, must
#   be an integer value
#Flag: tells which variable we're sampling for. Flag is a class of enums,
	# either Var.DELTA, Var.EPSILON, or Var.THETA
#Conducts Metropolis-Hastings simulation over specified number of
#   iterations: initializes a random value of delta/epsilon/theta, samples
#   from a Gaussian proposal distribution to propose a new value of
#   delta/epsilon/theta, and accepts that proposal depending on the posterior
#   probabilities of delta/epsilon/theta and the proposed new delta/epsilon/theta given in the
#   pdf_delta_epsilon/pdf_theta function
#   When sampling for theta, at each iteration MH will sample a theta list (a list of theta values for each verb), so at the end it will return a list of theta lists. 
#Returns vector of delta/epsilon/theta values after running specified number of iterations

import random
from pdf_delta_epsilon import pdf
from pdf_theta import pdf_theta_one_verb
from propose_and_accept import *



def MH(data, verb_categories, delta, epsilon, gammas, iterations, flag):

	if ((flag == Var.DELTA) or (flag == Var.EPSILON)):
		#Initialize a random value of epsilon/delta if sampling for one of those
		MHvar = random.random()
		sampled_results = [MHvar]
	else:
		#initialize a random value of theta for each verb if sampling for theta
		thetas = [random.random() for i in range(0, len(data))]
		sampled_results = [thetas]


		#Determine whether the variable being sampled with MH sampling is delta, epsilon, or theta
		#Use pdf to calculate logs of height of relevant variable on curve proportional to pdf over the sampled variable
	if flag == Var.DELTA:
		p_MHvar = pdf(data, verb_categories, MHvar, epsilon, gammas)
	elif flag == Var.EPSILON:
		p_MHvar = pdf(data, verb_categories, delta, MHvar, gammas)
	else:
		p_thetas = [pdf_theta_one_verb(data[j], delta, epsilon, thetas[j], gammas) for j in range(0, len(thetas))]


	for i in range(1, iterations):
		#note that this loop will actually run (iterations-1) times. This is because we want to consider
		#the initial random samples the 'first' iteration rather than the first time this loop runs.
		if ((flag == Var.DELTA) or (flag == Var.EPSILON)):
			#if we are using MH sampling for epsilon or delta, just call propose_and_accept one time per iteration
			result = propose_and_accept(data, verb_categories, delta, epsilon, gammas, MHvar, p_MHvar, flag)
			MHvar = result[0] #since propose_and_accept returns a tuple, set first element in tuple as MHvar
			p_MHvar = result[1] #set second element in tuple as p_MHvar
			#add the result to the sampled_results list
			sampled_results.append(MHvar)
		#this function returns a new MHvar
		else:
			#for theta, call propose_and_accept on each theta value (ie, for each verb)
			result = [propose_and_accept(data[j], verb_categories, delta, epsilon, gammas, thetas[j], p_thetas[j], flag) for j in range(0, len(thetas))]
			thetas = [i[0] for i in result] #propose_and_accept returns a list of tuples, set first element of each tuple as thetas list
			p_thetas = [i[1] for i in result] #set second element of each tuple as p_thetas list
			sampled_results.append(thetas) #add the accepted thetas to results list

	return sampled_results
