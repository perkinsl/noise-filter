#Calculates p(k|T,epsilon,delta), which is the likelihood of a given verb over three transitivity categories.
#With known epsilon and delta (a value between 0 and 1)
#verb: a 2-element list corresponding to counts of observations for each of n verbs. 
#   In each sublist, the first element contains counts of direct objects 
#   and the second contains total number of observations
#Delta: a value from 0 to 1
#Epsilon: a value from 0 to 1
#Gammas: dictionary of combination terms from binomial distribution equations,
#    passed on to each iteration of Gibbs sampling in joint_inference.py
#T1dict, T2dict, T3dict: dictionaries of p(k1|n1, T) for each verb over three verb categories
#Calculates the likelihoods of a verb over three verb categories:
#   1: verb is fully transitive (theta = 1)
#   2: verb is fully intransitive (theta = 0)
#   3: verb is mixed (theta sampled from Beta(1,1) uniform distribution)
#Returns a 3-element vector of likelihoods of the given verb over three verb categories

import math
import numpy as np
import itertools
from operator import add

def likelihoods(verb, delta, epsilon, gammas, T1dict, T2dict, T3dict):

    k = verb[0]
    n = verb[1]

    Tlikelihood = []

    ## likelihood p(k|T, epsilon, delta)
    ## create tuples containing all combinations of n1 in range (0, n+1) and k1 in range (0, k+1)
    ## equivalent to "for n1 in range (0, n+1) for k1 in range (0, k+1)"
    n1 = range(n+1)
    k1 = range(k+1)
    combinations = list(itertools.product(n1, k1)) ## returns cartesian product of n1 x k1

    ## implementing Equation (9) in Perkins, Feldman & Lidz: p(k0|n0, delta), in log space
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
    
    ## implementing Equation (10) in Perkins, Feldman & Lidz: p(k1|n1, T), in log space, for each verb category
    ## T1k1 is result for transitives (T=1)
    def calculate_T1k1(n1, k1):
        if (n1, k1) in T1dict:
            T1k1term = T1dict[(n1, k1)]

        else:
            if k1 <= n1:
                if k1 == n1:
                    T1k1term = 0 ## log of 1: transitive category has pr. 1 if k1 = n1
                else:
                    T1k1term = float('-inf') ## log of zero: transitive category has pr. 0 for all other k1, n1 combinations
            else:
                T1k1term = float('-inf')
            T1dict[(n1, k1)] = T1k1term

        return T1k1term
    
    ## T2k1 is result for intransitives (T=2)
    def calculate_T2k1(n1, k1):
        if (n1, k1) in T2dict:
           T2k1term = T2dict[(n1, k1)]

        else:
            if k1 <= n1:
                if k1 == 0:
                    T2k1term = 0 ## log of 1: intransitive category has pr. 1 if k1 = 0
                else:
                    T2k1term = float('-inf') ## log of zero: intransitive category has pr. 0 for all other k1, n1 combinations

            else:
                T2k1term = float('-inf')

            T2dict[(n1, k1)] = T2k1term

        return T2k1term
    
    ## T3k1 is result for alternators (T=3)
    def calculate_T3k1(n1, k1):
        if (n1, k1) in T3dict:
            T3k1term = T3dict[(n1, k1)]

        else:
            if k1 <= n1:
                T3k1term = np.log(1.0)-np.log(n1+1) ## result of integrating over all values of theta: 1/(n1+1)
            else:
                T3k1term = float('-inf')

            T3dict[(n1, k1)] = T3k1term

        return T3k1term
    
    #list of functions
    calculate_Tk1 = [calculate_T1k1, calculate_T2k1, calculate_T3k1]

    def calculate_T_likelihood(transitivity):

        Tcomponent = []
        
        ## group k1s by n1 values in order to efficiently compute inner sums in Equation (8)
        for key, group in itertools.groupby(combinations, lambda x: x[0]):
            ngroup = list(group)
            k0term = list(itertools.starmap(calculate_k0, ngroup))
            #choose which calculate_Mk1 function to call based on the transitivity category
            Tk1term = list(itertools.starmap(calculate_Tk1[transitivity], ngroup))

            Tterm = list(map(add, Tk1term, k0term))

            Tterm.sort(reverse=True)

            if Tterm[0] == float('-inf'):
                Ttermsub = Tterm

            else:
                Ttermsub = [(i-Tterm[0]) for i in Tterm]

            Ttermexp = [math.exp(i) for i in Ttermsub]

            Tlogsum = Tterm[0] + np.log1p(sum(Ttermexp[1:]))

            if (key, n) in gammas:
                noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)

            else:
                gammas[(key, n)] = math.lgamma(n+1)-(math.lgamma(key+1)+math.lgamma(n-key+1))
                noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)

            Tcomponent.append(Tlogsum + noise)

        Tcomponent.sort(reverse=True)

        if Tcomponent[0] == float('-inf'):
            Tcomponentsub = Tcomponent

        else:
            Tcomponentsub = [(i-Tcomponent[0]) for i in Tcomponent]

        Tcomponentexp = [math.exp(i) for i in Tcomponentsub]
        #instead of updating the value of T1likelihood, T2likelihood, T3likelihood, we return the calculated likekihood value
        return Tcomponent[0] + np.log1p(sum(Tcomponentexp[1:]))

    #calculate likelihood over three categories
    Tlikelihood = [calculate_T_likelihood(transitivity) for transitivity in range(3)]

    return Tlikelihood

