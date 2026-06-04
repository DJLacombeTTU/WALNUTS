import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pymc as pm
import numpy as np
import arviz as az
from sampler import sample_walnuts

def run_econometrics_test():
    # 1. Generate Dummy Economic Data
    np.random.seed(42)
    N = 100
    
    income = np.random.normal(0, 1, N) 
    
    true_beta = 2.5
    true_alpha = 10.0
    true_sigma = 1.0
    
    # Expenditure = Alpha + Beta * Income + Noise
    expenditure = true_alpha + true_beta * income + np.random.normal(0, true_sigma, N)
    
    print("Building PyMC Model...")
    with pm.Model() as model:
        # 1. Weakly Informative Priors (letting the likelihood dominate)
        alpha = pm.Normal("alpha", mu=0, sigma=20) 
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        
        # 2. Standardized Likelihood
        mu = alpha + beta * income
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=expenditure)
        
        # 3. Use sample_walnuts 
        idata = sample_walnuts(draws=2000, tune=2000, chains=4)

    # 3. Evaluate the results
    print("\n--- Posterior Means ---")
    
    alpha_mean = idata.posterior['alpha'].mean().item()
    beta_mean = idata.posterior['beta'].mean().item()
    
    print(f"Alpha (True: {true_alpha}): {alpha_mean:.3f}")
    print(f"Beta  (True: {true_beta}): {beta_mean:.3f}")
    
    if 'sigma_log__' in idata.posterior:
        sigma_mean = np.exp(idata.posterior['sigma_log__']).mean().item()
    else:
        sigma_mean = idata.posterior['sigma'].mean().item()
        
    print(f"Sigma (True: {true_sigma}): {sigma_mean:.3f}")

    print("\n--- ArviZ Summary ---")
    print(az.summary(idata, var_names=['alpha', 'beta']))

if __name__ == "__main__":
    run_econometrics_test()