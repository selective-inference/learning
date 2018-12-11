import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance
from selection.algorithms.lasso import ROSI

from learn_selection.core import (infer_full_target,
                                  split_sampler, # split_sampler not working yet
                                  normal_sampler,
                                  logit_fit,
                                  repeat_selection,
                                  probit_fit)
from learn_selection.keras_fit import keras_fit

def simulate(n=200, p=100, s=10, signal=(0.5, 1), sigma=2, alpha=0.1):

    # description of statistical problem

    X, y, truth = gaussian_instance(n=n,
                                    p=p, 
                                    s=s,
                                    equicorrelated=False,
                                    rho=0.5, 
                                    sigma=sigma,
                                    signal=signal,
                                    random_signs=True,
                                    scale=False)[:3]

    dispersion = sigma**2

    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    smooth_sampler = normal_sampler(S, covS)
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(XTX, XTXi, lam, sampler):

        p = XTX.shape[0]
        success = np.zeros(p)

        loss = rr.quadratic_loss((p,), Q=XTX)
        pen = rr.l1norm(p, lagrange=lam)

        scale = 0.5
        noisy_S = sampler(scale=scale)
        loss.quadratic = rr.identity_quadratic(0, 0, -noisy_S, 0)
        problem = rr.simple_problem(loss, pen)
        soln = problem.solve(max_its=100, tol=1.e-10)
        success += soln != 0
        return set(np.nonzero(success)[0])

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    lam = 4. * np.sqrt(n)
    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi, lam)

    # run selection algorithm

    success_params = (1, 1)

    observed_set = repeat_selection(selection_algorithm, smooth_sampler, *success_params)

    # find the target, based on the observed outcome

    # we just take the first target  

    pivots, covered, lengths, pvalues = [], [], [], []
    lower, upper = [], []
    naive_pvalues, naive_pivots, naive_covered, naive_lengths =  [], [], [], []

    R = ROSI.gaussian(X, y, lam, approximate_inverse=None)
    R.fit()
    summaryR = None
    #summaryR = R.summary(truth=truth[R.active], dispersion=dispersion, compute_intervals=True, level=1-alpha)
    #summaryR0 = R.summary(dispersion=dispersion, compute_intervals=False)
    #print(summaryR)
    #print(R.active, 'huh')

    targets = []
    true_target = truth[sorted(observed_set)]

    results = infer_full_target(selection_algorithm,
                                observed_set,
                                sorted(observed_set),
                                splitting_sampler,
                                dispersion,
                                hypothesis=true_target,
                                fit_probability=keras_fit,
                                fit_args={'epochs':20, 'sizes':[100]*5, 'dropout':0., 'activation':'relu'},
                                success_params=success_params,
                                alpha=alpha,
                                B=3000)
    for result in results:
        print(result)
        (pivot, 
         interval,
         pvalue,
         _) = result

        pvalues.append(pvalue)
        pivots.append(pivot)
        covered.append((interval[0] < true_target[0]) * (interval[1] > true_target[0]))
        lengths.append(interval[1] - interval[0])

    for idx in sorted(observed_set):
        target_sd = np.sqrt(dispersion * XTXi[idx, idx])
        observed_target = np.squeeze(XTXi[idx].dot(X.T.dot(y)))
        quantile = ndist.ppf(1 - 0.5 * alpha)
        naive_interval = (observed_target - quantile * target_sd, observed_target + quantile * target_sd)

        naive_pivot = (1 - ndist.cdf((observed_target - true_target[0]) / target_sd))
        naive_pivot = 2 * min(naive_pivot, 1 - naive_pivot)
        naive_pivots.append(naive_pivot)

        naive_pvalue = (1 - ndist.cdf(observed_target / target_sd))
        naive_pvalue = 2 * min(naive_pivot, 1 - naive_pivot)
        naive_pvalues.append(naive_pvalue)

        naive_covered.append((naive_interval[0] < true_target[0]) * (naive_interval[1] > true_target[0]))
        naive_lengths.append(naive_interval[1] - naive_interval[0])
        lower.append(interval[0])
        upper.append(interval[1])

    if summaryR is not None:
        liu_pivots = summaryR['pval']
        liu_pvalues = summaryR['pval']
        liu_lower = summaryR['lower_confidence']
        liu_upper = summaryR['upper_confidence']
        liu_lengths = liu_upper - liu_lower
        liu_covered = [(l < t) * (t < u) for l, u, t in zip(liu_lower, liu_upper, truth[R.active])]
    else:
        liu_pivots = liu_pvalues = liu_lower = liu_upper = liu_lengths = liu_covered = []

    if len(pvalues) > 0:
        return pd.DataFrame({'pivot':pivots,
                             'pvalue':pvalues,
                             'coverage':covered,
                             'length':lengths,
                             'naive_pivot':naive_pivots,
                             'naive_coverage':naive_covered,
                             'naive_length':naive_lengths,
                             #'liu_pivot':liu_pivots,
                             #'liu_pvalue':liu_pvalues,
                             #'liu_length':liu_lengths,
                             #'liu_upper':liu_upper,
                             #'liu_lower':liu_lower,
                             'upper':upper,
                             'lower':lower,
                             #'liu_coverage':liu_covered,
                             'target':truth[sorted(observed_set)]})


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    U = np.linspace(0, 1, 101)
    plt.clf()

    for i in range(500):
        df = simulate()
        csvfile = 'lasso_multi_random.csv'

        if df is not None and i > 0:

            try:
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass

            if len(df['pivot']) > 0:

                print("selective:", np.mean(df['pivot']), np.std(df['pivot']), np.mean(df['length']), np.std(df['length']), np.mean(df['coverage']))
                print("naive:", np.mean(df['naive_pivot']), np.std(df['naive_pivot']), np.mean(df['naive_length']), np.std(df['naive_length']), np.mean(df['naive_coverage']))

                print("len ratio selective divided by naive:", np.mean(np.array(df['length']) / np.array(df['naive_length'])))

                plt.clf()
                U = np.linspace(0, 1, 101)
                plt.plot(U, sm.distributions.ECDF(df['pivot'])(U), 'r', label='Selective', linewidth=3)
                plt.plot(U, sm.distributions.ECDF(df['naive_pivot'])(U), 'b', label='Naive', linewidth=3)
                #plt.plot(U, sm.distributions.ECDF(df['liu_pivot'][~np.isnan(df['liu_pivot'])])(U), 'g', label='Liu', linewidth=3)
                plt.legend()
                plt.plot([0,1], [0,1], 'k--', linewidth=2)
                plt.savefig(csvfile[:-4] + '.pdf')

                plt.clf()
                plt.scatter(df['naive_length'], df['length'])
                #plt.scatter(df['naive_length'], df['liu_length'])
                plt.savefig(csvfile[:-4] + '_lengths.pdf')

            df.to_csv(csvfile, index=False)

