import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance

from core import (infer_full_target,
                  split_sampler, # split_sampler not working yet
                  normal_sampler,
                  logit_fit,
                  probit_fit)

def simulate(n=1000, p=20, signal=3, sigma=2, alpha=0.1):

    # description of statistical problem

    X, y, truth = gaussian_instance(n=n,
                                    p=p, 
                                    s=5,
                                    equicorrelated=False,
                                    rho=0.5, 
                                    sigma=sigma,
                                    signal=signal,
                                    random_signs=True)[:3]

    dispersion = sigma**2

    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    smooth_sampler = normal_sampler(S, covS)
    splitting_sampler = split_sampler(X * y[:, None], covS / n)

    def meta_algorithm(XTX, XTXi, dispersion, lam, sampler):

        min_success = 2
        ntries = 5
        p = XTX.shape[0]
        success = np.zeros(p)

        loss = rr.quadratic_loss((p,), Q=XTX)
        pen = rr.l1norm(p, lagrange=lam)

        for _ in range(ntries):
            scale = 0.5
            noisy_S = sampler(scale=scale)
            loss.quadratic = rr.identity_quadratic(0, 0, -noisy_S, 0)
            problem = rr.simple_problem(loss, pen)
            soln = problem.solve(max_its=50, tol=1.e-6)
            success += soln != 0
        return set(np.nonzero(success >= min_success)[0])

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi, 2 * np.sqrt(n), dispersion)

    # run selection algorithm

    observed_set = selection_algorithm(splitting_sampler)

    # find the target, based on the observed outcome

    # we just take the first target  

    pivots, covered, lengths, naive_lengths = [], [], [], []
    for idx in observed_set:
        print(idx, len(observed_set))
        true_target = truth[idx]

        (pivot, 
         interval) = infer_full_target(selection_algorithm,
                                       observed_set,
                                       idx,
                                       splitting_sampler,
                                       dispersion,
                                       hypothesis=true_target,
                                       fitter=probit_fit,
                                       alpha=alpha,
                                       B=1000)

        pivots.append(pivot)
        covered.append((interval[0] < true_target) * (interval[1] > true_target))
        lengths.append(interval[1] - interval[0])

        target_sd = np.sqrt(dispersion) * XTXi[idx, idx]
        naive_lengths.append(2 * ndist.ppf(1 - 0.5 * alpha) * target_sd)

    return pivots, covered, lengths, naive_lengths

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    
    U = np.linspace(0, 1, 101)
    P, L, N, coverage = [], [], [], []
    plt.clf()
    for i in range(30):
        p, cover, l, n = simulate()
        coverage.extend(cover)
        P.extend(p)
        L.extend(l)
        N.extend(n)
        print(np.mean(P), np.std(P), np.mean(np.array(L) / np.array(N)), np.mean(coverage))

    plt.clf()
    plt.plot(U, sm.distributions.ECDF(P)(U), 'r', linewidth=3)
    plt.plot([0,1], [0,1], 'k--', linewidth=2)
    plt.savefig('lasso_example.pdf')
