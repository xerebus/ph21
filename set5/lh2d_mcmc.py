# Ph21 Set 5
# Aritra Biswas

# coin_mcmc.py
# Run MCMC on coin_model.py

import lh2d_model
from pymc import MCMC
from pymc.Matplot import plot

M = MCMC(lh2d_model)
M.sample(iter = 100, burn = 0, thin = 1)
print
plot(M)
M.alpha.summary()
M.beta.summary()
