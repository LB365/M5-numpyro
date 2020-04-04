import numpy as onp
from numpyro.diagnostics import autocorrelation, hpdi

class Metrics():

    def __init__(self,**kwargs):
        self.alpha = 0.95
        self._moments = None
        self._hit_rate = None
        self._data = None
        self._trace = None
        self.trace = None
        self.actual = None
        for name,value in kwargs.items():
            setattr(self,name,value)

    @property
    def moments(self):
        if self._moments is None:
            mean = onp.mean(self.trace, axis=0)
            hpdi_0, hpdi_1 = hpdi(self.trace, prob=self.alpha)
            names, values = ['lower', 'mean', 'upper'], [hpdi_0, mean, hpdi_1]
            self._moments = dict(zip(names, values))
        return self._moments

    @property
    def hit_rate(self):
        if self._hit_rate is None:
            hit = (self.actual >= self.moments['lower']) & (self.actual <= self.moments['upper'])
            self._hit_rate = hit.sum()/hit.size
        return self._hit_rate