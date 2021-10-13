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
#Epsilon: a decimal from 0 to 1
#Delta: a decimal from 0 to 1
#Returns p, height of function proportional to pdf of posterior probability
#   on epsilon/delta, at specified value of epsilon/delta


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
