
import torch
import numpy as np
from torch import nn
from scipy import linalg
from torch.nn import functional as F
from utils.operations import squeeze_nc, unsqueeze_nc
from custom_pkg.pytorch.operations import api_empty_cache


class TransposableModule(nn.Module):
    """
    Base class for linearized transposable modules.
    """
    def forward(self, *args, **kwargs):
        raise ValueError

    def linearized_transpose(self, eps):
        """ The Jacobian of this mapping equals J^\top, where J is the Jacobian of the forward mapping. """
        raise NotImplementedError


class InvertibleModule(TransposableModule):
    """
    Base class for invertible modules.
    """
    def inverse(self, *args, **kwargs):
        """ The inverted mapping of the forward mapping. """
        raise NotImplementedError


########################################################################################################################
# Convolutions & nonlinearities.
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Convolutions
# ----------------------------------------------------------------------------------------------------------------------

class InvConv2d1x1Fixed(InvertibleModule):
    """
    Invconv1x1 with fixed rotation matrix as the weight.
    """
    def __init__(self, input_nc):
        super(InvConv2d1x1Fixed, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Architecture
        # --------------------------------------------------------------------------------------------------------------
        matrix_r = torch.from_numpy(linalg.qr(np.random.randn(input_nc, input_nc))[0].astype('float32')).unsqueeze(-1).unsqueeze(-1)
        """ Set weight. """
        self.register_buffer("_matrix_r", matrix_r)

    def forward(self, x):
        return F.conv2d(x, self._matrix_r)

    def linearized_transpose(self, eps):
        return F.conv_transpose2d(eps, self._matrix_r)

    def inverse(self, x):
        return F.conv2d(x, self._matrix_r.transpose(0, 1))


class Conv2d(TransposableModule):
    """
    Conv layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(Conv2d, self).__init__()
        # Architecture.
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self._conv(x)

    def linearized_transpose(self, eps):
        return F.conv_transpose2d(eps, self._conv.weight, stride=self._conv.stride, padding=self._conv.padding)


# ----------------------------------------------------------------------------------------------------------------------
# Nonlinearities
# ----------------------------------------------------------------------------------------------------------------------

class ReLU(TransposableModule):
    """
    Element-wise ReLU activation.
    """
    def __init__(self):
        super(ReLU, self).__init__()
        # Gradient buffers.
        self._grads = None

    def forward(self, x, linearize=False):
        # Calculate output.
        output = torch.relu(x)
        """ Linearize. """
        if linearize: self._grads = torch.gt(x, torch.zeros_like(x)).to(x.dtype).detach()
        # Return
        return output

    @api_empty_cache
    def linearized_transpose(self, eps):
        # --------------------------------------------------------------------------------------------------------------
        """ Given x (n_x, ...) and grads (n_grads, ...), in the case of n_x > n_grads, the grads are repeatedly used.
        Namely it should be that n_x=n_grads*n_repeat, so grads should be repeated first. """
        grads = self._grads
        if len(eps) > len(grads):
            grads = grads.unsqueeze(1).expand(grads.size(0), len(eps)//len(grads), *grads.size()[1:]).reshape(*eps.size())
        # --------------------------------------------------------------------------------------------------------------
        # Calculate output
        output = eps * grads
        # Return
        return output


########################################################################################################################
# Utils.
########################################################################################################################

class Squeeze(InvertibleModule):
    """
    Squeeze Fn.
    """
    def __init__(self, s=2):
        super(Squeeze, self).__init__()
        # Config.
        self._s = s

    def forward(self, x):
        return squeeze_nc(x, s=self._s)

    def linearized_transpose(self, eps):
        return unsqueeze_nc(eps, s=self._s)

    def inverse(self, x):
        return unsqueeze_nc(x, s=self._s)


class Unsqueeze(InvertibleModule):
    """
    Unsqueeze Fn.
    """
    def __init__(self, s=2):
        super(Unsqueeze, self).__init__()
        # Config.
        self._s = s

    def forward(self, x):
        return unsqueeze_nc(x, s=self._s)

    def linearized_transpose(self, eps):
        return squeeze_nc(eps, s=self._s)

    def inverse(self, x):
        return squeeze_nc(x, s=self._s)
