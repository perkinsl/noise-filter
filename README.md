# noise-filter
Bayesian verb argument structure learner with noise filter

This repo contains all scripts for running the model simulations reported in Perkins, Feldman, & Lidz (under review), "The Power of Ignoring: Filtering Input for Argument Structure Acquisition." Scripts were written in Python 3 by Laurel Perkins, Xinyue Cui, and Shalinee Maitra. Contact: perkinsl@ucla.edu

Notes: all probabilities in these scripts are in log space except where comments indicate otherwise. The dataset compiled from the CHILDES Treebank (Pearl & Sprouse, 2013) is summarized in Perkins, Feldman, & Lidz (under revision). Runtime for these scripts is quite long over this dataset (several hours to several days depending on processor). Scripts can also be tested in the mini toy datasets provided in Test_data.xlsx.

----------------------------------------------------------------
SCRIPTS INCLUDE:

- jointinference.py: performs Gibbs sampling over 1000 number of iterations. Within each iteration, it jointly infers verb transitivity categories (T, aka "verb_categories") by calling sample_categories.py as well as noise filter parameters (epsilon and delta). Epsilon and delta are sampled with a Metropolis-Hastings proposal by calling MH.py. Outputs .txt files listing every 10th value of T, epsilon, and delta from the last 500 iterations of Gibbs sampling as samples from the posterior distributions over those variables. Additionally outputs .png files plotting distributions over epsilon and delta, and a .txt file summarizing counts of transitivity categories sampled for each verb ("category_table").

- MH.py: performs specified number of iterations of Metropolis-Hastings sampling for either epsilon, delta, or theta, depending on which variable needs to be sampled. Calls pdf and pdf_theta_one verb to initialize values for variables. Then, calls propose_and_accept to propose a new value at each iteration, sampled from a Gaussian with mu set to the previous variable value and sigma = 0.25. Accepts with probability f(new value)/f(old value), where f is a function returning a value proportional to the posterior probability on epsilon, delta, or theta. Note that when sampling for theta, the function samples a value for one particular verb rather than the entire dataset. When sampling for theta, the function assumes a list of alternating verbs only. It samples for the theta value of each verb at a time, and returns a list of the accepted results for each verb in the data structure. 

- pdf_theta.py: calculates f(x) for specific value x of theta, where f is a function returning a value proportional to the posterior probability on theta. Note that this file works on one verb at a time, and the MH.py script calls it on each verb in the data structure. 

- pdf_delta_epsilon.py: calculates f(x) for specific value x of epsilon or delta, where f is a function returning a value proportional to the posterior probability on epsilon or delta. Calls likelihoods.py to do most of the calculations.

- sample_categories.py: calculates posterior probability for each transitivity category T for each verb in dataset, given epsilon and delta. Called in each iteration of joint_inference. 

- propose_and_accept.py:  Calls the pdf_theta_one_verb and pdf functions to propose a new value sampled from a Gaussian with mu set to previous value and sigma = 0.25. Accepts with probability f(new value)/f(old value), where f is a function returning a value proportional to the posterior probability on epsilon and delta. 

- likelihoods.py: calculates p(k|T,epsilon,delta), which is the likelihood of a given verb's data under three transitivity categories.

Dependencies: joint_inference.py imports MH.py and sample_categories.py. MH.py imports pdf_theta.py, pdf_delta_epsilon.py, and propose_and_accept.py. sample_categories.py imports likelihoods.py. pdf_theta.py has no dependencies on other scripts. pdf_delta_epsilon imports likelihoods.py. propose_and_accept imports pdf_theta.py and pdf_delta_epsilon.py.  


-----------------------------------------------------------------
INSTRUCTIONS FOR RUNNING SCRIPTS:

To run the basic joint inference script on the original dataset, make sure all of these files are located in the same directory, and then call joint_inference.py.

To change the dataset for the model, comment out the data vector at the end of joint_inference.py and paste in your own data vector. Or better yet, modify the script to read in a .csv or .txt data file.

The number of iterations for Gibbs Sampling and Metropolis-Hastings sampling can be changed by changing the relevant arguments to the functions in joint_inference.py.

The priors on transitivity categories (T) can be adjusted in sample_models.py.

-----------------------------------------------------------------
EXAMPLE:

Here's an example of what you should see if you run the model with a small toy dataset.

In joint_inference, comment out the data vector, and then let
data = [[19, 20], [9, 10], [1, 20], [2, 40], [10, 20], [3, 10]] 

In this example, we also modified the code to run a shorter Gibbs sampling chain: we ran Gibbs sampling only 20 times and saved the results from the last 10 iterations.

If you then run joint_inference.py, the output should look something like:

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

The output describes the sampled verb categories of the 6 verbs during each iteration.

The list of 3-element lists at the end indicate the total number of occurrences of the 3 categories for each verb, which is saved to a separate .txt file named category_table. Note that the output prints out the categories sampled in all 20 iterations, but we only take the samples from the last 10 iterations in computing the final transitivity category.

To determine the transitivity category that the model assigned to each verb, Perkins, Feldman & Lidz chose the category with highest posterior probability. For this data set, the transitivity category assigned to each verb by the model would be [1, 1, 2, 2, 3, 3], representing "transitive, transitive, intransitive, intransitive, alternating, alternating."

The output also includes two histograms of the samples of delta and epsilon, which are saved to separate .png files named delta.png and epsilon.png. These visualize the inferred posterior probability distributions over these variables. For this dataset, the distribution over delta is centered at 0.5, and the distribution over epsilon is centered at 0.15.

The dataset was generated with an actual delta of 0.5, an actual epsilon of 0.1, and actual verb categories of [1, 1, 2, 2, 3, 3], which match our result.
