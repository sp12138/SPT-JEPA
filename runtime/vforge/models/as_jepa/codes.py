import math

def kl_const(sigma0):                    # per-dimension constant of KL(N(mu, sigma0^2) || N(0, 1))
    return 0.5 * sigma0 ** 2 - 0.5 - math.log(sigma0)

def entry_nats(mu0, sigma0):             # mu0: [B, K, d] -> KL(N(mu0, sigma0^2 I) || N(0, I)) per window, [B]
    mu0 = mu0.float()
    return 0.5 * mu0.pow(2).sum((-2, -1)) + mu0[0].numel() * kl_const(sigma0)

def surprise_nats(mu, m, sigma0):        # mu, m: [B, T, K, d] -> KL(N(mu, sigma0^2 I) || N(m, sigma0^2 I)), [B, T]
    return (mu.float() - m.float()).pow(2).sum((-2, -1)) / (2 * sigma0 ** 2)
