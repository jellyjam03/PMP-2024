import scipy.stats as stats
import arviz as az

y = 7
n = 10

alpha_prior = 1
beta_prior = 1

alpha_posterior = alpha_prior + y
beta_posterior = beta_prior + n - y

posterior_dist = stats.gamma(a=alpha_posterior, scale=1/beta_posterior)

az.plot_posterior(posterior_samples, hdi_prob=0.94)