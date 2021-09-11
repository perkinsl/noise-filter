#Simulates a learner encountering a corpus of observations of many verbs (data)
#With known epsilon and delta (a value between 0 and 1)
#Data: a list of length n where each item is a 2-element list corresponding
#   to counts of observations for each of n verbs. In each sublist, the first element
#   contains counts of direct objects and the second contains total number of observations
#Gammas: dictionary of combination terms from binomial distribution equations,
#    passed on to each iteration of Gibbs sampling in joint_inference.py
#Infers posterior probabilities on models (aka, verb transitivity classes) for each verb in data:
#    M1: verb is fully transitive (theta = 1)
#    M2: verb is fully intransitive (theta = 0)
#    M3: verb is mixed (theta sampled from Beta(1,1) uniform distribution)
#Samples a model value for each verb by flipping a biased coin weighted by
#   those posterior probabilities over models
#Returns a vector of model values (1, 2, or 3) for each verb in the data
#   matrix, where each element in vector corresponds to a row in the data
#   matrix

#Updates 08/03/2021: replaced repeated code with function likelihoods

#Updates 08/10/2021: got rid of the for loop within function sample_models

#Updates 09/08/2021: added helper function proportionate_model_posterior to reduce repetitive code

import math
import random
import numpy as np
from pdf_delta_epsilon import likelihoods

# calculating denominator for Equation (7)
def proportionate_model_posterior(transitivity, verbLikelihoods):
    # prior P(T) from Equation (7) is flat: 1/3 for each value of T
    Mprior = 1.0/3.0
    demM = verbLikelihoods[transitivity] + np.log(Mprior)
    return demM

def calculate_model(verbNumber, data, epsilon, delta, gammas, M1dict, M2dict, M3dict):

    verbcount = data[verbNumber]

    verbLikelihoods = likelihoods(verbcount, delta, epsilon, gammas, M1dict, M2dict, M3dict)

    #keeping this intermediate step here because it's needed when calculating models_posterior
    dems = [proportionate_model_posterior(i, verbLikelihoods) for i in range(3)]

    demsexp = [math.exp(dem) for dem in dems] # list comprehension: [fun(i) for i in list]
    denominator = np.log(sum(demsexp))

    # final result of Equation (7), in log space
    #no need to calculate M3 since the prediction will automatically fall into M3 if it doesn't fit into M1 and M2
    models_posterior = [(dems[i] - denominator) for i in range(2)]

    #flips a biased coin weighted by posteriors on models to sample a model (ie, category) for each verb
    x = random.random()
    if x <= math.exp(models_posterior[0]):
        return 1

    elif x <= (math.exp(models_posterior[0]) + math.exp(models_posterior[1])):
        return 2

    else:
        return 3

def sample_models(data, epsilon, delta, gammas):
	verb_categories = []

	## memoizing specific n1, k1 combinations for Equation (10) in Perkins, Feldman & Lidz
	## because these will always produce the same result, regardless of the verb identity
	M1dict = {}
	M2dict = {}
	M3dict = {}

	## loop through every verb in dataset and calculate posterior on transitivity models (T)
	## following Equation (7) in Perkins, Feldman, & Lidz
	verb_categories = [calculate_model(verb, data, epsilon, delta, gammas, M1dict, M2dict, M3dict) for verb in range(len(data))]

	return verb_categories
