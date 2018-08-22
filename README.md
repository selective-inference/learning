Some methods to learn selection probability.

Pseudocode
----------

For simple selection procedures, this pseudocode describes
our approach

```
import numpy as np
from copy import copy
from selection.distributions.discrete_family import discrete_family
import rpy2.robjects as rpy

# description of statistical problem

truth = np.array([1,2.])

data = np.random.standard_normal((100, 2)) + np.multiply.outer((100,), truth) / np.sqrt(100)

def sufficient_stat(data):
    return np.mean(data, 0)

S = sufficient_stat(data)

# randomization mechanism

class normal_sampler(object):

    def __init__(self, center, covariance):
        (self.center,
         self.covariance) = (np.asarray(center),
                             np.asarray(covariance))
        self.cholT = np.linalg.cholesky(self.covariance)
        self.shape = self.center.shape

    def __call__(self, scale=1., size=None):

        if type(size) == type(1):
            size = (size,)
        size = size or (1,)
        return np.sqrt(scale) * np.squeeze(np.random.standard_normal(size + self.shape).dot(self.cholT))

    def __copy__(self):
        return normal_sampler(self.center.copy(),
                              self.covariance.copy())

observed_sampler = normal_sampler(S, 1/100. * np.identity(2))   

assert(observed_sampler(size=(200,)).shape == (200, 2)) # 200 iid draws N(S, sigma) (given S)

def algo_constructor():

    def myalgo(sampler):
        noisyS = sampler(scale=0.25)
        return np.fabs(noisyS.sum()) > 2

# run selection algorithm

algo_instance = algo_constructor()
observed_outcome = algo_instance(observed_sampler())

# find the target, based on the observed outcome

def compute_target(observed_outcome, data):
    if observed_outcome: # target is mu[0]
        observed_target, target_cov, cross_cov = sufficient_stat(data)[0], 1/100. * np.identity(1), np.array([1., 0.])
    else:
        observed_target, target_cov, cross_cov = sufficient_stat(data)[1], 1/100. * np.identity(1), np.array([0., 1.])
    return observed_target, target_cov, cross_cov

observed_target, target_cov, cross_cov = compute_target(observed_outcome, data)
direction = cross_cov.dot(np.linalg.inv(target_cov))

def learning_proposal():
    scale = np.random.choice([0.5, 1, 2, 3], 1)
    return np.random.standard_normal() * scale

def logit_fit(T, Y):
    rpy.r.assign('T', T)
    rpy.r.assign('Y', Y)
    rpy.r('''
    M = glm(Y ~ T, family=binomial(link='logit'))
    fitfn = function(t) { predict(M, newdata=data.frame(T=t), type='link') }
    ''')

    fitfn = rpy.r('fitfn')
    return fitfn

def probit_fit(T, Y):
    rpy.r.assign('T', T)
    rpy.r.assign('Y', Y)
    rpy.r('''
    M = glm(Y ~ T, family=binomial(link='probit'))
    fitfn = function(t) { predict(M, newdata=data.frame(T=t), type='link') }
    ''')

    fitfn = rpy.r('fitfn')
    return fitfn

def learn_weights(algorithm, learning_proposal, fit_probability, B=15000):

    new_sampler = copy(normal_sampler)

    learning_sample = []
    for _ in range(1000):
         T = learning_proposal()      # a guess at informative distribution for learning what we want
         new_sampler.center = S + direction * (T - observed_target)
         Y = algorithm(new_sampler()) == observed_outcome

    conditional_law = fit_probability(T, Y)
    return conditional_law

weight_fn = learn_weights(algorithm, learning_proposal, probit_fit)

# let's form the pivot

if observed_outcome:
    true_target = mu[0]
else:
    true_target = mu[1]

target_sampler =  normal_sampler(0, target_cov)
target_sample = normal_sampler(size=(5000,))
weights_sample = weight_fn(target_sample)

# for p == 1 targets this is what we do -- have some code for multidimensional too

exp_family = discrete_family(target_sample, weights_sample)  
pivot = exp_family.cdf(true_target, observed_target)
interval = exp_family.equal_tailed_interval(observed_target) 
```
