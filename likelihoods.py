#Simulates a learner encountering a corpus of observations of many verbs (data)
#With known epsilon and delta (a value between 0 and 1)
#verb: a 2-element list corresponding to counts of observations for each of n verbs. 
#   In each sublist, the first element contains counts of direct objects 
#   and the second contains total number of observations
#Delta: a value from 0 to 1
#Epsilon: a value from 0 to 1
#Gammas: dictionary of combination terms from binomial distribution equations,
#    passed on to each iteration of Gibbs sampling in joint_inference.py
#Calculates the likelihoods of a verb over three verb categories:
#   1: verb is fully transitive (theta = 1)
#   2: verb is fully intransitive (theta = 0)
#   3: verb is mixed (theta sampled from Beta(1,1) uniform distribution)
#Returns a 3-element vector of likelihoods of the given verb over three verb categories

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

    #did not define this function outside of function likelihoods because functions calculate_M1k1, calculate_M2k1, calculate_M3k1
    #are defined within function likelihoods and our function need to call those
    def calculate_M_likelihood(transitivity):

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
        #instead of updating the value of M1likelihood, M2likelihood, M3likelihood, we return the calculated likekihood value
        return Mcomponent[0] + np.log1p(sum(Mcomponentexp[1:]))

    #calculate likelihood over three categories
    Mlikelihood = [calculate_M_likelihood(transitivity) for transitivity in range(3)]

    return Mlikelihood

