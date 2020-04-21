import scipy.stats


def pct2sigma(pct):
    return scipy.stats.norm.ppf(pct)

def sigma2pct(sigma):
    return scipy.stats.norm.cdf(sigma)