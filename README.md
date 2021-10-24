# noise-filter
Bayesian verb argument structure learner with noise filter

This repo contains all scripts for running the model simulations reported in Perkins, Feldman, & Lidz (in revision), "The Power of Ignoring: Filtering Input for Argument Structure Acquisition." Scripts were written in Python 3. Copyright Laurel Perkins; contact: perkinsl@ucla.edu.

Notes: all probabilities in these scripts are in log space except where comments indicate otherwise. The dataset compiled from the CHILDES Treebank (Pearl & Sprouse, 2013) is summarized in Perkins, Feldman, & Lidz (under revision). Runtime for these scripts is quite long over this dataset (several hours to several days depending on processor). Scripts can also be tested in the mini toy datasets provided in Test_data.xlsx.

----------------------------------------------------------------
SCRIPTS INCLUDE:

- jointinference.py: performs Gibbs sampling over 1000 number of iterations (calls sample_categories.py within each iteration) to jointly infer verb transitivity categories (T, aka "verb_categories") as well as filter parameters (epsilon and delta), with 10 rounds of Metropolis-Hastings sampling for epsilon and delta within each Gibbs sampling iteration (Calls MH.py to do this). Outputs .txt files listing every 10th value of T, epsilon, and delta from last 500 iterations as samples from the posterior distributions over those variables. Additionally outputs .png files plotting distributions over epsilon and delta, and a .txt file summarizing counts of transitivity categories sampled for each verb ("category_table").

- MH.py: perform specified number of iterations of Metropolis-Hastings sampling for either epsilon, delta, or theta, depending on which variable one needs to sample. Uses different flag for each variable. For delta the flag should be 0, for epsilon flag should be 1, for theta flag should be 2. Calls pdf and pdf_theta_one verb to initialize values for variables. Then, calls propose_and_accept to propose a new value at each iteration, sampled from a Gaussian with mu set to previous value and sigma = 0.25. Accepts with probability f(new value)/f(old value), where f is a function returning a value proportional to the posterior probability on epsilon and delta. Note that flag 2, which samples for theta, is for a version of this model that assumes only one verb category–alternating verbs. It calls the file pdf_theta, and it samples for the theta value of each verb (as opposed to epsilon and delta, which are sampled over the entire dataset). Sampling for theta can be used with a list of alternating verbs. Returns a list of the accepted results. 

- pdf_theta.py: calculates f(x) for specific value x of theta, where f is a function returning a value proportional to the posterior probability on theta. Note that this file works on one verb at a time, and the MH.py script with flag 2 calls it on each verb in the data structure. 

- pdf_delta_epsilon.py: calculates f(x) for specific value x of epsilon or delta, where f is a function returning a value proportional to the posterior probability on epsilon or delta. Calls likelihoods.py to do most of the calculations.

- sample_categories.py: calculates posterior probability for each transitivity category T for each verb in dataset, given epsilon and delta. Called in each iteration of joint_inference. 

- propose_and_accept.py:  Calls the pdf_theta_one_verb and pdf functions to propose a new value sampled from a Gaussian with mu set to previous value and sigma = 0.25. Accepts with probability f(new value)/f(old value), where f is a function returning a value proportional to the posterior probability on epsilon and delta. 

- likelihoods.py: calculates p(k|T,epsilon,delta), which is the likelihood of a given verb over three transitivity categories.

Dependencies: joint_inference.py imports MH.py and sample_categories.py. MH.py imports pdf_theta.py, pdf_delta_epsilon.py, and propose_and_accept.py. sample_categories.py imports likelihoods.py. pdf_theta.py has no dependencies on other scripts. pdf_delta_epsilon imports likelihoods.py. propose_and_accept imports pdf_theta.py and pdf_delta_epsilon.py.  


-----------------------------------------------------------------
INSTRUCTIONS FOR RUNNING SCRIPTS:

To run the basic joint inference script on the original dataset, make sure all of these files are located in the same directory, and then call joint_inference.py.

To change the dataset for the model, comment out the data vector at the end of joint_inference.py and paste in your own data vector. Or better yet, modify the script to read in a .csv or .txt data file.

The number of iterations for Gibbs Sampling and Metropolis-Hastings sampling can be changed by changing the relevant arguments to the functions in joint_inference.py.

The priors on transitivity categories (T) can be adjusted in sample_models.py.

To examine the effects of randomly varying the values for epsilon and delta, call test_epsilon_delta.py. To examine the effects of randomly varying the prior over T, call test_priors.py.

We can test joint_inference with a smaller toy data set.
We comment out the data vector, then let
data = [[19, 20], [9, 10], [1, 20], [2, 40], [10, 20], [3, 10]] 
and run joint_inference.py.
Note here we also replaced the code for running Gibbs sampling 1000 iterations and saving every tenth result in the last 500 iterations
with the code for running Gibbs sampling 20 times and saving the results from the last 10 iterations for the sake of demonstration.
The output should look something like:

iteration 0
categories [1, 2, 2, 2, 2, 2]
iteration 1
categories [1, 1, 2, 2, 3, 2]
iteration 2
categories [1, 1, 2, 2, 3, 2]
iteration 3
categories [1, 1, 2, 2, 1, 3]
iteration 4
categories [1, 1, 2, 2, 3, 3]
iteration 5
categories [1, 3, 2, 3, 3, 3]
iteration 6
categories [1, 1, 2, 2, 3, 3]
iteration 7
categories [1, 1, 2, 2, 3, 3]
iteration 8
categories [1, 3, 2, 2, 3, 3]
iteration 9
categories [1, 1, 2, 2, 3, 3]
iteration 10
categories [1, 1, 2, 2, 3, 3]
iteration 11
categories [1, 3, 2, 2, 3, 3]
iteration 12
categories [1, 1, 2, 3, 3, 2]
iteration 13
categories [1, 3, 2, 2, 3, 3]
iteration 14
categories [1, 1, 2, 2, 3, 3]
iteration 15
categories [1, 1, 2, 2, 3, 3]
iteration 16
categories [1, 3, 2, 2, 3, 3]
iteration 17
categories [1, 1, 3, 2, 3, 3]
iteration 18
categories [1, 1, 2, 2, 3, 3]
iteration 19
categories [1, 1, 2, 2, 3, 3]
[[10  0  0]
 [ 7  0  3]
 [ 0  9  1]
 [ 0  9  1]
 [ 0  0 10]
 [ 0  1  9]]

The output describes the predicted verb categories of the 6 verbs during each iteration.
The last 6 lines indicate the total number of occurrences of the 3 categories for each verb, which are saved to a separate txt file named category_table.
Note that the output prints out the categories sampled in all 20 iterations but we only take the samples from the last 10 iterations in computing the final transitivity category.
The transitivity category is determined by the category that was assigned the highest probability to each verb by our model described in Perkins, Feldman, & Lidz, so for this data set, the transitivity category for each verb by our model should be [1, 1, 2, 2, 3, 3].
Additionally, the output also includes two histograms of the distribution over delta and epsilon, which are saved to separate png files named delta.png and epsilon.png, respectively.
For this dataset, the distribution over delta is centered at 0.5, and the distribution over epsilon is centered at 0.15.
The actual delta of this data set = 0.5, and actual epsilon = 0.1, actual models = [1, 1, 2, 2, 3, 3], which match our result.
