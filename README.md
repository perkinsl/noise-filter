# noise-filter
Bayesian verb argument structure learner with noise filter

This repo contains all scripts for running the model simulations reported in Perkins, Feldman, & Lidz (in revision), "The Power of Ignoring: Filtering Input for Argument Structure Acquisition." Scripts were written in Python 3. Copyright Laurel Perkins; contact: perkinsl@ucla.edu.

Notes: all probabilities in these scripts are in log space except where comments indicate otherwise. The dataset compiled from the CHILDES Treebank (Pearl & Sprouse, 2013) is summarized in Perkins, Feldman, & Lidz (under revision). Runtime for these scripts is quite long over this dataset (several hours to several days depending on processor). Scripts can also be tested in the mini toy datasets provided in Test_data.xlsx.

----------------------------------------------------------------
SCRIPTS INCLUDE:

- jointinference.py: performs Gibbs sampling over 1000 number of iterations to jointly infer verb transitivity categories (T, aka verb "models") as well as filter parameters (epsilon and delta), with 10 rounds of Metropolis-Hastings sampling for epsilon and delta within each Gibbs sampling iteration. Outputs .txt files listing every 10th value of T, epsilon, and delta from last 500 iterations as samples from the posterior distributions over those variables. Additionally outputs .png files plotting distributions over epsilon and delta, and a .txt file summarizing counts of transitivity categories sampled for each verb ("modeltable").


- MH.py: perform specified number of iterations of Metropolis-Hastings sampling for either epsilon, delta, or theta, depending on which variable one needs to sample. Uses different flag for each variable. For delta the flag should be 0, for epsilon flag should be 1, for theta flag should be 2.  Proposes a new value at each iteration, sampled from a Gaussian with mu set to previous value and sigma = 0.25. Accepts with probability f(new value)/f(old value), where f is a function returning a value proportional to the posterior probability on epsilon and delta.

- pdf_delta_epsilon: calculates f(x) for specific value x of epsilon or delta, where f is a function returning a value proportional to the posterior probability on epsilon or delta. Calls likelihoods.py to do most of the calculations.

- sample_models.py: calculates posterior probability for each transitivity category T for each verb in dataset, given epsilon and delta.

- propose_and_accept.py: proposes a new value sampled from a Gaussian with mu set to previous value and sigma = 0.25. Accepts with probability f(new value)/f(old value), where f is a function returning a value proportional to the posterior probability on epsilon and delta.

- likelihoods.py: calculates p(k|T,epsilon,delta), which is the likelihood of a given verb over three transitivity categories.

- oracle_model.py: calculates posterior probability for each transitivity category T for each verb in dataset, given estimated oracle values for epsilon and delta. No Gibbs sampling is needed in this case, so this script bypasses those sampling steps. Outputs a .txt file summarizing these probabilities ("oraclemodeltable").

- test_epsilon_delta.py: tests importance of specific filter parameters by randomly sampling 500 epsilon, delta pairs and inferring highest-probability transitivity category for each verb from the posterior probability over T given these parameter settings. Outputs a .txt file containing sampled epsilon, delta, and highest-probability category for each verb ("test_epsilon_delta_modeltable").

- test_priors.py: tests importance of prior over T by randomly sampling 500 prior values for each category of T. Infers highest-probability transitivity category for each verb form the posterior probability over T given these priors and the mean values for the filter model parameters epsilon and delta inferred by the joint inference learner. Outputs a .txt file containing sampled priors and the highest-probability category for each verb ("test_priors_modeltable").

-----------------------------------------------------------------
INSTRUCTIONS FOR RUNNING SCRIPTS:

To run the basic joint inference script on the original dataset, make sure all of these files are located in the same directory, and then call joint_inference.py.

To change the dataset for the model, comment out the data vector at the end of joint_inference.py and paste in your own data vector. Or better yet, modify the script to read in a .csv or .txt data file.

The number of iterations for Gibbs Sampling and Metropolis-Hastings sampling can be changed by changing the relevant arguments to the functions in joint_inference.py.

The priors on transitivity categories (T) can be adjusted in sample_models.py.

To run the oracle model (best estimate of epsilon and delta from dataset), call oracle_model.py. Other specific values of epsilon and delta can be tested by changing the relevant arguments in this script.

To examine the effects of randomly varying the values for epsilon and delta, call test_epsilon_delta.py. To examine the effects of randomly varying the prior over T, call test_priors.py.
