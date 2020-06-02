"""
Classes for generating random noise.
"""

import numpy as np
# TODO: pass seed and init a separate PRG for each noise?

class Noise():
    """Interface that each Noise generator should implement"""
    def __init__(self):
        pass

    def get(self, shape):
        """Generate noise of given `shape`

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the noise to be generated.

        Returns
        -------
        ndarray, shape `shape`
            Noise of given shape
        """
        raise NotImplementedError

class DebugNoise(Noise):
    """Noise that returns predefined values.

    Parameters
    ----------
    values : list of float
        List of values to be returned instead of random noise.
        C.f. `DebugNoise.get`
    """

    def __init__(self, values):
        super(DebugNoise, self).__init__()
        self._values = np.array(values)

    def get(self, shape):
        if isinstance(shape, list):
            return self._values[:shape[0]]
        else:
            return self._values[:shape]

class UniformNoise(Noise):
    """Noise sampled from a continuous uniform distribution.

    Parameters
    ----------
    a : float
        Low edge of a uniform distribution from which noise will be sampled.
    b : float
        High edge of a uniform distribution from which noise will be sampled.
    """

    def __init__(self, a, b):
        super(UniformNoise, self).__init__()
        self._a = a
        self._b = b

    def get(self, shape):
        return np.random.uniform(self._a, self._b, shape)

class DiscreteNoise(Noise):
    """Noise sampled from a discrete distribution.

    Parameters
    ----------
    values : list of float
        A set of values from which noise will be sampled.
    prob : list of float or None, optional
        Probability for each value in `values`. If None, all values are assumed
        to have equal probability.
    """

    def __init__(self, values, prob = None):
        super(DiscreteNoise, self).__init__()
        self._values = values
        self._prob   = prob

    def get(self, shape):
        return np.random.choice(
            self._values, size = shape, replace = True, p = self._prob
        )

class GaussianNoise(Noise):
    """Noise sampled from a normal distribution.

    Parameters
    ----------
    mu : float
        Mean of the normal distribution, from which noise is to be sampled.
    sigma : float
        Standard deviation of the normal distribution, from which noise is to
        be sampled.
    """

    def __init__(self, mu, sigma):
        super(GaussianNoise, self).__init__()
        self._mu    = mu
        self._sigma = sigma

    def get(self, shape):
        return np.random.normal(self._mu, self._sigma, shape)

def select_noise(name, **kwargs):
    """Constructs `Noise` class based on a name

    Parameters
    ----------
    name : { None, 'uniform', 'gaussian', 'discrete', 'debug' }
        Type of the `Noise` to be returned.
    **kwargs : dict
        Noise parameters to be passed to the `Noise` constructor.
    """

    if name is None:
        return None

    if name == 'uniform':
        return UniformNoise(**kwargs)

    if name == 'gaussian':
        return GaussianNoise(**kwargs)

    if name == 'discrete':
        return DiscreteNoise(**kwargs)

    if name == 'debug':
        return DebugNoise(**kwargs)

    raise ValueError("Unknown noise: %s" % name)

