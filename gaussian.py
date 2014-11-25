## Tonya Benjamin
## C463

## Gaussian Mixture (Normal Distribution)
## modeled off of code by Thomas Rueckstiess

#! /usr/bin/python
import pybrain
import scipy
from scipy import random
from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.tools.functions import expln         
# expln ensures values of array are always postive. 
# Returns exp(val) for x>=0. Returns ln(x+1)+1 for x<0. 
from pybrain.tools.functions import explnPrime
# explnPrime is first deritivative of self.expln

from pybrain.structure.parametercontainer import ParameterContainer


class GaussianLayer(NeuronLayer, ParameterContainer):
    """ A layer implementing a gaussian (Normal) interpretation of the input. The mean is the input, the sigmas are stored in the module parameters."""

    def __init__(self, dim, name=None):
        NeuronLayer.__init__(self, dim, name)
        # initialize sigmas to 0
        ParameterContainer.__init__(self, dim, stdParams = 0)
        # if autoalpha is set to True, alpha_sigma = alpha_mu = alpha*sigma^2
        self.autoalpha = False
        self.enabled = True


    def setSigma(self, sigma):
        """Wrapper method to set the sigmas (the parameters of the module) to a certain value. """
        assert len(sigma) == self.indim
        self._params *= 0
        self._params += sigma


    def _forwardImplementation(self, inbuf, outbuf):
        if not self.enabled:
            outbuf[:] = inbuf
        else:
            outbuf[:] = random.normal(inbuf, expln(self.params))


    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        expln_params = expln(self.params)
        self._derivs += ((outbuf - inbuf)**2 - expln_params**2) / expln_params * explnPrime(self.params)
        inerr[:] = (outbuf - inbuf)

        if not self.autoalpha:
            inerr /= expln_params**2
            self._derivs /= expln_params**2
