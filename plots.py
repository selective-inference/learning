import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


U = np.linspace(0, 1, 101)
file_labels = ['cv_probit3.csv', 'cv_logit3.csv']

dfs = {}
for label in zip(file_labels, ['logit', 'probit']):
    print(label[1])
    dfs[label[1]] = pd.read_csv(label[0])
    (coverage, 
     P, 
     L, 
     naive_coverage, 
     naive_P, 
     naive_L) = (dfs[label[1]]['coverage'],
                 dfs[label[1]]['pval'],
                 dfs[label[1]]['length'],
                 dfs[label[1]]['naive_coverage'],
                 dfs[label[1]]['naive_pval'],
                 dfs[label[1]]['naive_length'])
    
    print("selective:", np.mean(P), np.std(P), np.mean(L), np.mean(coverage))
    print("naive:", np.mean(naive_P), np.std(naive_P), np.mean(naive_L), np.mean(naive_coverage))
    print("len ratio selective divided by naive:", np.mean(np.array(L) / np.array(naive_L)))


probit_P, naive_P = dfs['probit']['pval'], dfs['probit']['naive_pval']
logit_P = dfs['logit']['pval']


plt.clf()
plt.plot(U, sm.distributions.ECDF(probit_P)(U), 'c', linewidth=3, label = "fit probit")
plt.plot(U, sm.distributions.ECDF(logit_P)(U), 'b', linewidth=3, label="fit logit")
plt.plot(U, sm.distributions.ECDF(naive_P)(U), 'y', linewidth=3, label="naive")
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel("Observed pivot", fontsize=18)
plt.ylabel("Proportion (empirical CDF)", fontsize=18)
plt.title("Pivots", fontsize=20)
plt.legend(fontsize=18, loc="lower right")
plt.savefig('cv_pivots.pdf')
