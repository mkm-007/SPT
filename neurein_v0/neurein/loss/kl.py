"""
kl.py — KL Divergence loss for VAEs

KL( q(z|x) || N(0,I) ) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

This is the closed-form KL divergence between a diagonal Gaussian
q(z|x) = N(mu, sigma^2) and the standard normal prior N(0, I).

In the VAE loss:
  Total loss = Reconstruction loss + beta * KL

The KL term forces the encoder to produce latent distributions
that are close to the standard normal — this is what gives the
latent space its smooth, organised structure.
"""

from neurein.nn.module import Module


class KLDivergenceLoss(Module):
    """
    KL divergence between encoder distribution and N(0,I).

    Args:
        mu:      mean of encoder distribution, shape (batch, latent_dim)
        log_var: log variance of encoder distribution, shape (batch, latent_dim)
        beta:    weight on KL term (beta=1 is standard VAE, beta>1 is beta-VAE)
    """

    def forward(self, mu, log_var, beta=1.0):
        # -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp())
        return kl.mean() * beta
