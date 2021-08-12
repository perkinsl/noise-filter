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
#Delta: a decimal from 0 to 1
#Returns p, height of function proportional to pdf of posterior probability
#   on epsilon, at specified value of epsilon

#Updates 07/29/2021: merged pdf_delta and pdf_epsilon (which are identical except for the order of delta and epsilon in the input argument) into one file and updated variable names in MH_delta and MH_epsilon

#Updates 08/03/2021: create function likelihoods (identical to original pdf_delta and pdf_epsilon function from line 43 to line 198), which calculates the M1likehood,
#    M2likelihood, M3likelihood of a given verb. Here, M1dict, M2dict, M3dict are passed in as arguments because we need to save these values globally, just like gammas. 

#Updates 08/10/2021: got rid of the for loop within function pdf

#Updates 08/11/2021: write the computation of verb likelihoods over three transitivity categories in a for loop

import math
import numpy as np
import itertools
from operator import add


def likelihoods(verb, delta, epsilon, gammas, M1dict, M2dict, M3dict):
    
    k = verb[0]
    n = verb[1]
    
    Mlikelihood = []
    
    ## likelihood p(k|T, epsilon, delta)
    ## create tuples containing all combinations of n1 in range (0, n+1) and k1 in range (0, k+1)
    ## equivalent to "for n1 in range (0, n+1) for k1 in range (0, k+1)"
    n1 = range(n+1)
    k1 = range(k+1)
    combinations = list(itertools.product(n1, k1))
    
    def calculate_k0(n1, k1):
        if (k-k1)<=(n-n1):
            if ((k-k1), (n-n1)) in gammas:
                k0term = gammas[((k-k1), (n-n1))]+(k-k1)*np.log(delta)+((n-n1)-(k-k1))*np.log(1-delta)
            else:
                gammas[((k-k1), (n-n1))] = math.lgamma(n-n1+1)-(math.lgamma(k-k1+1)+math.lgamma((n-n1)-(k-k1)+1))
                k0term = gammas[((k-k1), (n-n1))]+(k-k1)*np.log(delta)+((n-n1)-(k-k1))*np.log(1-delta)
        
        else:
            k0term = float('-inf')
        
        return k0term

    def calculate_M1k1(n1, k1):
        if (n1, k1) in M1dict:
            M1k1term = M1dict[(n1, k1)]
        
        else:
            if k1 <= n1:
                if k1 == n1:
                    M1k1term = 0
                else:
                    M1k1term = float('-inf')
            else:
                M1k1term = float('-inf')
            M1dict[(n1, k1)] = M1k1term
    
        return M1k1term

    def calculate_M2k1(n1, k1):
        if (n1, k1) in M2dict:
            M2k1term = M2dict[(n1, k1)]
        
        else:
            if k1 <= n1:
                if k1 == 0:
                    M2k1term = 0
                else:
                    M2k1term = float('-inf')
        
            else:
                M2k1term = float('-inf')
    
            M2dict[(n1, k1)] = M2k1term
                        
        return M2k1term

    def calculate_M3k1(n1, k1):
        if (n1, k1) in M3dict:
            M3k1term = M3dict[(n1, k1)]
        
        else:
            if k1 <= n1:
                M3k1term = np.log(1.0)-np.log(n1+1)
            else:
                M3k1term = float('-inf')
            
            M3dict[(n1, k1)] = M3k1term
        
        return M3k1term
    #list of functions
    calculate_Mk1 = [calculate_M1k1, calculate_M2k1, calculate_M3k1]
    
    for transitivity in range(3):
    
        Mcomponent = []
        ## group by n1s
        for key, group in itertools.groupby(combinations, lambda x: x[0]):
            ngroup = list(group)
            k0term = list(itertools.starmap(calculate_k0, ngroup))
            #choose which calculate_Mk1 function to call based on the transitivity category
            Mk1term = list(itertools.starmap(calculate_Mk1[transitivity], ngroup))
        
            Mterm = list(map(add, Mk1term, k0term))
        
            Mterm.sort(reverse=True)
        
            if Mterm[0] == float('-inf'):
                Mtermsub = Mterm
        
            else:
                Mtermsub = [(i-Mterm[0]) for i in Mterm]
    
            Mtermexp = [math.exp(i) for i in Mtermsub]
        
            Mlogsum = Mterm[0] + np.log1p(sum(Mtermexp[1:]))
        
            if (key, n) in gammas:
                noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)

            else:
                gammas[(key, n)] = math.lgamma(n+1)-(math.lgamma(key+1)+math.lgamma(n-key+1))
                noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)

            Mcomponent.append(Mlogsum + noise)

        Mcomponent.sort(reverse=True)

        if Mcomponent[0] == float('-inf'):
            Mcomponentsub = Mcomponent
    
        else:
            Mcomponentsub = [(i-Mcomponent[0]) for i in Mcomponent]

        Mcomponentexp = [math.exp(i) for i in Mcomponentsub]
        #instead of updating the value of M1likelihood, M2likelihood, M3likelihood, we append the calculated likekihood value to the list of Mlikelihood
        Mlikelihood.append(Mcomponent[0] + np.log1p(sum(Mcomponentexp[1:])))

    return Mlikelihood

def likelihood_given_M(verbNumber, data, models, delta, epsilon, gammas, M1dict, M2dict, M3dict):
    
    verbcount = data[verbNumber]
    #given a verb, calculates the likelihoods over three categories
    verbLikelihoods = likelihoods(verbcount, delta, epsilon, gammas, M1dict, M2dict, M3dict)
        
    if models[verbNumber] == 1:
        #verbposteriors.append(verbLikelihoods[0])
        return verbLikelihoods[0]
            
    elif models[verbNumber] == 2:
        #verbposteriors.append(verbLikelihoods[1])
        return verbLikelihoods[1]
    
    elif models[verbNumber] == 3:
        #verbposteriors.append(verbLikelihoods[2])
        return verbLikelihoods[2]
            
    else:
        print('Invalid model value')
        return float('-inf')

def pdf(data, models, delta, epsilon, gammas):
    
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
        verbposteriors = [likelihood_given_M(verb, data, models, delta, epsilon, gammas, M1dict, M2dict, M3dict) for verb in range(len(models))]
        
        #for i in range(len(verbLikelihood)):
            #verbposteriors.append(verbLikelihood[i])
## function g(delta) in Equation (13) is equal to product across all verbs of likelihood term, times prior on delta
## prior is equal to 1 for all values of delta, because delta ~ Beta(1,1),
## so this reduces to product across all verbs of likelihood term
## and here, we're returning that value in log space
        p = sum(verbposteriors)

    return p
