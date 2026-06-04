import pymc as pm
import numpy as np
import arviz as az
import time
from sampler import sample_walnuts

def test_neals_funnel(draws=2000, tune=2000, chains=4):
    print("\n" + "="*50)
    print("TEST 1: NEAL'S FUNNEL (MULTISCALE GEOMETRY)")
    print("="*50)
    
    with pm.Model() as funnel_model:
        # The Funnel Axis (Group Variance)
        y = pm.Normal('y', mu=0, sigma=3)
        # The Funnel Mouth/Neck (Lower-level parameters)
        x = pm.Normal('x', mu=0, sigma=pm.math.exp(y / 2), shape=10)
        
        t0 = time.time()
        idata = sample_walnuts(draws=draws, tune=tune, chains=chains)
        t1 = time.time()
        
    print(f"Sampling Time: {t1 - t0:.2f} seconds")
    print(az.summary(idata, var_names=['y']))
    return idata

def test_collinear_logit(draws=2000, tune=2000, chains=4):
    print("\n" + "="*50)
    print("TEST 2: COLLINEAR BAYESIAN LOGIT (MISCONDUCT PROXY)")
    print("="*50)
    
    np.random.seed(42)
    N = 500
    
    # Highly correlated, unstandardized variables
    base_income = np.random.uniform(50000, 150000, N)
    flexible_income = base_income + np.random.normal(0, 5000, N)
    corporate_exp = base_income * 0.4 + np.random.normal(0, 10000, N)
    
    # True Log-Odds
    true_logits = -5.0 + 0.0001 * flexible_income - 0.0002 * corporate_exp
    probabilities = 1 / (1 + np.exp(-true_logits))
    misconduct_obs = np.random.binomial(1, probabilities)
    
    with pm.Model() as logit_model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta_inc = pm.Normal('beta_inc', mu=0, sigma=10)
        beta_exp = pm.Normal('beta_exp', mu=0, sigma=10)
        
        logits = alpha + beta_inc * flexible_income + beta_exp * corporate_exp
        y_obs = pm.Bernoulli('y_obs', logit_p=logits, observed=misconduct_obs)
        
        t0 = time.time()
        idata = sample_walnuts(draws=draws, tune=tune, chains=chains)
        t1 = time.time()
        
    print(f"Sampling Time: {t1 - t0:.2f} seconds")
    print(az.summary(idata))
    return idata

def test_hierarchical_spatial(draws=2000, tune=2000, chains=4):
    print("\n" + "="*50)
    print("TEST 3: HIERARCHICAL POOLING (SPATIAL PROXY)")
    print("="*50)
    
    np.random.seed(42)
    regions = 25
    N_per_region = 20
    
    region_idx = np.repeat(np.arange(regions), N_per_region)
    true_regional_effects = np.random.normal(5.0, 2.0, regions)
    y_val = true_regional_effects[region_idx] + np.random.normal(0, 1.0, regions * N_per_region)
    
    with pm.Model() as hierarchical_model:
        # Hyperpriors
        mu_region = pm.Normal('mu_region', mu=0, sigma=10)
        sigma_region = pm.HalfNormal('sigma_region', sigma=5)
        
        # Regional spatial effects
        regional_offsets = pm.Normal('regional_offsets', mu=0, sigma=1, shape=regions)
        regional_effects = mu_region + regional_offsets * sigma_region
        
        # Likelihood
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=5)
        y_obs = pm.Normal('y_obs', mu=regional_effects[region_idx], sigma=sigma_obs, observed=y_val)
        
        t0 = time.time()
        idata = sample_walnuts(draws=draws, tune=tune, chains=chains)
        t1 = time.time()
        
    print(f"Sampling Time: {t1 - t0:.2f} seconds")
    print(az.summary(idata, var_names=['mu_region', 'sigma_region', 'sigma_obs']))
    return idata

if __name__ == "__main__":
    print("Initializing WALNUTS Engine Stress Test...")
    idata_funnel = test_neals_funnel()
    idata_logit = test_collinear_logit()
    idata_hierarchical = test_hierarchical_spatial()