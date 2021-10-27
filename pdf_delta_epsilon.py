#Calculates f(x) for specific value x of either epsilon or delta, where f is a function returning a value proportional to the posterior probability 
#   on epsilon or delta. Calls likelihoods.py to do most of the calculations.
#With known categories for verbs that generated that data:
#   1: verb is fully transitive (theta = 1)
#   2: verb is fully intransitive (theta = 0)
#   3: verb is mixed (theta sampled from Beta(1,1) uniform distribution)
#verbNumber: number corresponding to the verb's index in the 'data' vector
#Data: a list of length n where each item is a 2-element list corresponding
#   to counts of observations for each of n verbs. In each sublist, the first element
#   contains counts of direct objects and the second contains total number of observations
#verb_categories: a list of model values (1, 2, or 3) for each verb in the data
#   array, where each element in list corresponds to a row in the data
#   array
#Delta: a decimal from 0 to 1
#Epsilon: a decimal from 0 to 1
#Gammas: Memoization dictionary for likelihoods initialized in joint_inference.py
#T1dict, T2dict, T3dict: dictionaries of p(k1|n1, T) for each verb over three verb categories
#Returns p, height of function proportional to pdf of posterior probability
#   on epsilon/delta, at specified value of epsilon/delta


from likelihoods import likelihoods

def likelihood_given_T(verbNumber, data, verb_categories, delta, epsilon, gammas, T1dict, T2dict, T3dict):

    #verbNumber: index of each verb
    verbcount = data[verbNumber]
    #given a verb, calculates the likelihoods over three categories
    verbLikelihoods = likelihoods(verbcount, delta, epsilon, gammas, T1dict, T2dict, T3dict)

    if verb_categories[verbNumber] == 1:
        return verbLikelihoods[0]

    elif verb_categories[verbNumber] == 2:
        return verbLikelihoods[1]

    elif verb_categories[verbNumber] == 3:
        return verbLikelihoods[2]

    else:
        print('Invalid verb category value')
        return float('-inf')

def pdf(data, verb_categories, delta, epsilon, gammas):

    if delta < 0 or epsilon < 0:
        p = float('-inf')
    elif delta > 1 or epsilon > 1:
        p = float('-inf')
    else:
        verbposteriors = []

        T1dict = {}
        T2dict = {}
        T3dict = {}

        ## loop through every verb in dataset and calculate p(k|T,epsilon,delta)
        ## following likelihood function in Equation (8) in Perkins, Feldman & Lidz
        verbposteriors = [likelihood_given_T(verb, data, verb_categories, delta, epsilon, gammas, T1dict, T2dict, T3dict) for verb in range(len(verb_categories))]

## function f(x) (where X is epsilon or delta) is equal to product across all verbs of likelihood term, times prior on X
## prior is equal to 1 for all values of X, because epsilon and delta are both drawn from a Beta(1,1),
## so this reduces to product across all verbs of likelihood term only
## and here, we're returning that value in log space
        p = sum(verbposteriors)

    return p
