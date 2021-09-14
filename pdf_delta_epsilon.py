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
#Delta: a decimal from 0 to 1
#Returns p, height of function proportional to pdf of posterior probability
#   on epsilon, at specified value of epsilon

#Updates 07/29/2021: merged pdf_delta and pdf_epsilon (which are identical except for the order of delta and epsilon in the input argument) into one file and updated variable names in MH_delta and MH_epsilon

#Updates 08/03/2021: create function likelihoods (identical to original pdf_delta and pdf_epsilon function from line 43 to line 198), which calculates the M1likehood,
#    M2likelihood, M3likelihood of a given verb. Here, M1dict, M2dict, M3dict are passed in as arguments because we need to save these values globally, just like gammas.

#Updates 08/10/2021: got rid of the for loop within function pdf

#Updates 08/11/2021: write the computation of verb likelihoods over three transitivity categories in a for loop and later got rid of the for loop

#Updates 09/13/2021: import likelihoods function created in a separate file

from likelihoods import likelihoods

def likelihood_given_M(verbNumber, data, verb_categories, delta, epsilon, gammas, M1dict, M2dict, M3dict):

    verbcount = data[verbNumber]
    #given a verb, calculates the likelihoods over three categories
    verbLikelihoods = likelihoods(verbcount, delta, epsilon, gammas, M1dict, M2dict, M3dict)

    if verb_categories[verbNumber] == 1:
        #verbposteriors.append(verbLikelihoods[0])
        return verbLikelihoods[0]

    elif verb_categories[verbNumber] == 2:
        #verbposteriors.append(verbLikelihoods[1])
        return verbLikelihoods[1]

    elif verb_categories[verbNumber] == 3:
        #verbposteriors.append(verbLikelihoods[2])
        return verbLikelihoods[2]

    else:
        print('Invalid model value')
        return float('-inf')

def pdf(data, verb_categories, delta, epsilon, gammas):

    if delta < 0 or epsilon < 0:
        p = float('-inf')
    elif delta > 1 or epsilon > 1:
        p = float('-inf')
    else:
        verbposteriors = []

        M1dict = {}
        M2dict = {}
        M3dict = {}

        ## loop through every verb in dataset and calculate p(k|T,epsilon,delta)
        ## following likelihood function in Equation (8) in Perkins, Feldman & Lidz
        verbposteriors = [likelihood_given_M(verb, data, verb_categories, delta, epsilon, gammas, M1dict, M2dict, M3dict) for verb in range(len(verb_categories))]

        #for i in range(len(verbLikelihood)):
            #verbposteriors.append(verbLikelihood[i])
## function g(delta) in Equation (13) is equal to product across all verbs of likelihood term, times prior on delta
## prior is equal to 1 for all values of delta, because delta ~ Beta(1,1),
## so this reduces to product across all verbs of likelihood term
## and here, we're returning that value in log space
        p = sum(verbposteriors)

    return p
