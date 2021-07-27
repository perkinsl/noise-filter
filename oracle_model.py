#Simulates a learner encountering a corpus of observations of many verbs (data)
#With known epsilon and delta (a value between 0 and 1)
#Data: a list of length n where each item is a 2-element list corresponding 
#   to counts of observations for each of n verbs. In each sublist, the first element 
#   contains counts of direct objects and the second contains total number of observations
#Returns a matrix of posterior probabilities on models for each verb in data:
#    M1: verb is fully transitive (theta = 1)
#    M2: verb is fully intransitive (theta = 0)
#    M3: verb is mixed (theta sampled from Beta(1,1) uniform distribution)


import math
import random
import numpy as np
import itertools
from operator import add

def infer_model_probs(data, epsilon, delta, gammas):
	models = []

	M1dict = {}
	M2dict = {}
	M3dict = {}
	
	for i in range(0, len(data)):
		#calculates log posterior probabilies on transitivity models for each verb in the dataset
		
		verbcounts = data[i]
		
		M1prior = 1.0/3.0
		M2prior = 1.0/3.0
		M3prior = 1.0/3.0
		
		k = verbcounts[0]
		n = verbcounts[1]
		
		M1component = []
		M2component = []
		M3component = []
		
		## likelihood p(k|T, epsilon, delta)
		## create tuples containing all combinations of n1 in range (0, n+1) and k1 in range (0, k+1)
		## equivalent to "for n1 in range (0, n+1) for k1 in range (0, k+1)"
		n1 = range(n+1)
		k1 = range(k+1)
		combinations = list(itertools.product(n1, k1))
		
		def calculate_k2(n1, k1):
			if (k-k1)<=(n-n1):
				if ((k-k1), (n-n1)) in gammas:
					k2term = gammas[((k-k1), (n-n1))]+(k-k1)*np.log(delta)+((n-n1)-(k-k1))*np.log(1-delta)
				else:
					gammas[((k-k1), (n-n1))] = math.lgamma(n-n1+1)-(math.lgamma(k-k1+1)+math.lgamma((n-n1)-(k-k1)+1))
					k2term = gammas[((k-k1), (n-n1))]+(k-k1)*np.log(delta)+((n-n1)-(k-k1))*np.log(1-delta)
			
			else:
				k2term = float('-inf')
			
			return k2term
		
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
		
		## group by n1s
		for key, group in itertools.groupby(combinations, lambda x: x[0]):
			ngroup = list(group)
			k2term = list(itertools.starmap(calculate_k2, ngroup))
			M1k1term = list(itertools.starmap(calculate_M1k1, ngroup))
			M2k1term = list(itertools.starmap(calculate_M2k1, ngroup))
			M3k1term = list(itertools.starmap(calculate_M3k1, ngroup))
			
			M1term = list(map(add, M1k1term, k2term))
			M2term = list(map(add, M2k1term, k2term))
			M3term = list(map(add, M3k1term, k2term))
			
			M1term.sort(reverse=True)
			M2term.sort(reverse=True)
			M3term.sort(reverse=True)
		
			if M1term[0] == float('-inf'):
				M1termsub = M1term
			
			else:
				M1termsub = [(i-M1term[0]) for i in M1term]
			
			if M2term[0] == float('-inf'):
				M2termsub = M2term

			else:
				M2termsub = [(i-M2term[0]) for i in M2term]
			
			if M3term[0] == float('-inf'):
				M3termsub = M3term
			
			else:
				M3termsub = [(i-M3term[0]) for i in M3term]

			M1termexp = [math.exp(i) for i in M1termsub]
			M2termexp = [math.exp(i) for i in M2termsub]
			M3termexp = [math.exp(i) for i in M3termsub]
		
			M1logsum = M1term[0] + np.log1p(sum(M1termexp[1:]))
			M2logsum = M2term[0] + np.log1p(sum(M2termexp[1:]))
			M3logsum = M3term[0] + np.log1p(sum(M3termexp[1:]))
			
			if (key, n) in gammas:
				noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)
				
			else:
				gammas[(key, n)] = math.lgamma(n+1)-(math.lgamma(key+1)+math.lgamma(n-key+1))
				noise = gammas[(key, n)]+key*math.log(1-epsilon)+(n-key)*math.log(epsilon)
			
			M1component.append(M1logsum + noise)
			M2component.append(M2logsum + noise)
			M3component.append(M3logsum + noise)
		
		M1component.sort(reverse=True)
		M2component.sort(reverse=True)
		M3component.sort(reverse=True)
		
		if M1component[0] == float('-inf'):
			M1componentsub = M1component
		
		else:
			M1componentsub = [(i-M1component[0]) for i in M1component]
		
		if M2component[0] == float('-inf'):
			M2componentsub = M2component
		
		else:
			M2componentsub = [(i-M2component[0]) for i in M2component]
		
		if M3component[0] == float('-inf'):
			M3componentsub = M3component

		else:
			M3componentsub = [(i-M3component[0]) for i in M3component]

		M1componentexp = [math.exp(i) for i in M1componentsub]
		M2componentexp = [math.exp(i) for i in M2componentsub]
		M3componentexp = [math.exp(i) for i in M3componentsub]
		
		M1likelihood = M1component[0] + np.log1p(sum(M1componentexp[1:]))
		M2likelihood = M2component[0] + np.log1p(sum(M2componentexp[1:]))
		M3likelihood = M3component[0] + np.log1p(sum(M3componentexp[1:]))
		
		demM1 = M1likelihood + np.log(M1prior)
		demM2 = M2likelihood + np.log(M2prior)
		demM3 = M3likelihood + np.log(M3prior)
		
		dems = [demM1, demM2, demM3]
		demsexp = [math.exp(i) for i in dems]
		denominator = np.log(sum(demsexp))
		
		M1 = (M1likelihood + np.log(M1prior)) - denominator
		M2 = (M2likelihood + np.log(M2prior)) - denominator
		M3 = (M3likelihood + np.log(M3prior)) - denominator
		
		models.append([math.exp(M1), math.exp(M2), math.exp(M3)])
	
	modeltable = np.asarray(models)
	np.savetxt('oraclemodeltable', modeltable)
			
	return models

#data containing vector of verb counts from CHILDES Treebank
#first list element for each verb in the vector is counts of overt direct objects,
#and second list element is total count for that verb.
#see 'CHILDESTreebank_VerbData' for all 50 verbs in order.
data = [[308,1568], [777,1318], [11,859], [541,605], [3,605], [155,583], [406,579], [347,550], [350,509], [350,485], [287,477], [13,451], [57,383], [193,375], [221,366], [299,358], [297,356], [274,352], [265,342], [305,337], [299,331], [268,331], [275,312], [4,308], [161,306], [215,299], [21,294], [114,281], [8,275], [198,263], [11,256], [132,255], [11,253], [112,238], [13,228], [49,227], [205,220], [187,214], [8,197], [161,195], [57,192], [140,191], [141,185], [160,185], [153,183], [7,180], [149,169], [141,166], [115,160], [53,151]]

gammas = {}

print(infer_model_probs(data, 0.24, 0.18, gammas))
