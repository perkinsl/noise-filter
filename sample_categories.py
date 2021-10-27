#Simulates a learner encountering a corpus of observations of many verbs (data)
#With known epsilon and delta (a value between 0 and 1)
#verbNumber: number corresponding to the verb's index in the 'data' vector
#Data: a list of length n where each item is a 2-element list corresponding
#   to counts of observations for each of n verbs. In each sublist, the first element
#   contains counts of direct objects and the second contains total number of observations
#Delta: a decimal from 0 to 1
#Epsilon: a decimal from 0 to 1
#Gammas: dictionary of combination terms from binomial distribution equations,
#    passed on to each iteration of Gibbs sampling in joint_inference.py
#transitivity: integers representing verb categories, including transitive (1), intransitive (2), and alternating (3)
#Infers posterior probabilities on categories for each verb in data:
#    T1: verb is fully transitive (theta = 1)
#    T2: verb is fully intransitive (theta = 0)
#    T3: verb is mixed (theta sampled from Beta(1,1) uniform distribution)
#verbLikelihoods: likelihoods of each verb over three categories
#T1dict, T2dict, T3dict: dictionaries of p(k1|n1, T) for each verb over three verb categories
#Samples a category value for each verb by flipping a biased coin weighted by
#   those posterior probabilities over categories
#Returns a vector of category values (1, 2, or 3) for each verb in the data
#   matrix, where each element in vector corresponds to a row in the data
#   matrix


import math
import random
import numpy as np
from likelihoods import likelihoods

#Calculates numerator for Equation (7)
def proportionate_category_posterior(transitivity, verbLikelihoods):
    # prior P(T) from Equation (7) is flat: 1/3 for each value of T
    Tprior = 1.0/3.0
    numeratorT = verbLikelihoods[transitivity-1] + np.log(Tprior)
    return numeratorT

#Samples a verb category for the verb given
def calculate_category(verbNumber, data, epsilon, delta, gammas, T1dict, T2dict, T3dict):

    verbcount = data[verbNumber]

    verbLikelihoods = likelihoods(verbcount, delta, epsilon, gammas, T1dict, T2dict, T3dict)
	
    numerators = [proportionate_category_posterior(i, verbLikelihoods) for i in range(1,4)]

    numeratorsexp = [math.exp(i) for i in numerators]
    denominator = np.log(sum(numeratorsexp))

    # final result of Equation (7), in log space
    #no need to calculate T3 since the prediction will automatically fall into T3 if it doesn't fit into T1 and T2
    categories_posterior = [(numerators[i] - denominator) for i in range(2)]

    #flips a biased coin weighted by posteriors on categories to sample a category for each verb
    x = random.random()
    if x <= math.exp(categories_posterior[0]):
        return 1

    elif x <= (math.exp(categories_posterior[0]) + math.exp(categories_posterior[1])):
        return 2

    else:
        return 3

#Samples verb categories for all verbs
def sample_categories(data, epsilon, delta, gammas):
	verb_categories = []

	## memoizing specific n1, k1 combinations for Equation (10) in Perkins, Feldman & Lidz
	## because these will always produce the same result, regardless of the verb identity
	T1dict = {}
	T2dict = {}
	T3dict = {}

	## loop through every verb in dataset and calculate posterior on transitivity categories (T)
	## following Equation (7) in Perkins, Feldman, & Lidz
	verb_categories = [calculate_category(verb, data, epsilon, delta, gammas, T1dict, T2dict, T3dict) for verb in range(len(data))]    

	return verb_categories
