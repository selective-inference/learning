import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance

from learn_selection.utils import full_model_inference, pivot_plot
from learn_selection.core import normal_sampler, logit_fit
from learn_selection.learners import mixture_learner
mixture_learner.scales = [1]*10 + [1.5,2,3,4,5,10]

def BHfilter(pval, q=0.2):
    pval = np.asarray(pval)
    pval_sort = np.sort(pval)
    comparison = q * np.arange(1, pval.shape[0] + 1.) / pval.shape[0]
    passing = pval_sort < comparison
    if passing.sum():
        thresh = comparison[np.nonzero(passing)[0].max()]
        return np.nonzero(pval <= thresh)[0]
    return []

def simulate(n=200, p=30, s=10, signal=(0.5, 1), sigma=2, alpha=0.1, B=1000):

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

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    smooth_sampler = normal_sampler(S, covS)

    def meta_algorithm(XTX, XTXi, dispersion, sampler):

        p = XTX.shape[0]
        success = np.zeros(p)

        scale = 0.
        noisy_S = sampler(scale=scale)
        solnZ = noisy_S / (np.sqrt(np.diag(XTX)) * np.sqrt(dispersion))
        pval = ndist.cdf(solnZ)
        pval = 2 * np.minimum(pval, 1 - pval)
        return set(BHfilter(pval, q=0.2))

    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi, dispersion)

    # run selection algorithm

    return full_model_inference(X,
                                y,
                                truth,
                                selection_algorithm,
                                smooth_sampler,
                                success_params=(1, 1),
                                B=B,
                                fit_probability=logit_fit,
                                fit_args={'df':20})

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    for i in range(500):
        df = simulate(B=5000)
        csvfile = 'logit_targets_BH_marginal.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try: # concatenate to disk
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                pivot_ax, length_ax = pivot_plot(df, outbase)


