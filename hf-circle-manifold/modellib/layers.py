
from utils.operations import *
from modellib.components import *


########################################################################################################################
# Padding.
########################################################################################################################

class Padding(InvertibleModule):
    """
    The padding & projection layer.
    """
    def __init__(self, x_nc, u_nc):
        super(Padding, self).__init__()
        assert x_nc > u_nc
        # Config.
        self._u_nc, self._v_nc = u_nc, x_nc-u_nc

    @property
    def u_nc(self):
        return self._u_nc

    @property
    def v_nc(self):
        return self._v_nc

    def forward(self, u, v=None):
        """ Padding. """
        if v is None: v = torch.zeros(u.size(0), self._v_nc, *u.size()[2:]).type_as(u)
        """ Set uv. """
        return torch.cat([u, v], dim=1)

    def inverse(self, x):
        """ Projection. """
        u, v = x[:, :self._u_nc], x[:, self._u_nc:]
        # Return
        return u, v


########################################################################################################################
# Flow.
########################################################################################################################

class Logdet(object):
    """ Logdet wrapper. """
    def __init__(self, total_logdet, nc, img_size):
        # Config.
        self._nc, self._img_size = nc, img_size
        # Value.
        self._total_logdet = total_logdet

    def __add__(self, other):
        assert isinstance(other, Logdet)
        assert other._nc == self._nc and other._img_size == self._img_size
        # Return
        return Logdet(self._total_logdet+other._total_logdet, nc=self._nc, img_size=self._img_size)

    @property
    def total(self):
        return self._total_logdet

    @property
    def elem(self):
        return self._total_logdet / (self._nc*self._img_size**2)


class AffineCoupling(InvertibleModule):
    """
    Coupling: (nc, h, w) -> (nc, h, w).
    """
    def __init__(self, input_nc, hidden_nc):
        super(AffineCoupling, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        self._nn = nn.ModuleList([])
        # Hidden neural network.
        self._nn.append(Conv2d(input_nc//2, hidden_nc, kernel_size=3, stride=1, padding=1))
        self._nn.append(ReLU())
        self._nn.append(Conv2d(hidden_nc, hidden_nc, kernel_size=3, stride=1, padding=1))
        self._nn.append(ReLU())
        self._nn.append(Conv2d(hidden_nc, input_nc, kernel_size=3, stride=1, padding=1))
        # --------------------------------------------------------------------------------------------------------------
        # Gradients buffer
        # --------------------------------------------------------------------------------------------------------------
        self._grads = {'x_b': None, 's': None, 't': None}

    def forward(self, x, linearize=False, compute_logdet=False):
        # --------------------------------------------------------------------------------------------------------------
        # Forward.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Split.
        x_a, x_b = x.chunk(2, dim=1)
        # 2. Coupling.
        # (1) NN.
        output_nn = x_a
        for module in self._nn:
            kwargs = {'linearize': linearize} if isinstance(module, ReLU) else {}
            # Forward.
            output_nn = module(output_nn, **kwargs)
        # (2) Affine.
        logs, t = output_nn.chunk(2, dim=1)
        s = logs.exp()
        output_b = (x_b + t) * s
        # 3. Merge.
        output = torch.cat([x_a, output_b], dim=1)
        """ Linearize """
        if linearize: self._grads = {'x_b': x_b, 's': s, 't': t}
        # --------------------------------------------------------------------------------------------------------------
        # Compute logdet.
        # --------------------------------------------------------------------------------------------------------------
        if compute_logdet:
            logdet = Logdet(total_logdet=torch.sum(logs, dim=(1, 2, 3)), nc=x.size(1), img_size=x.size(2))
            # Return
            return output, logdet
        # Return
        return output

    def linearized_transpose(self, eps):
        """ Given x_a -> output_b as output_b = (x_b+t) * s, we have that
            output_b' = (x_b+t)*s' + t'*s = (x_b+t)*s*logs' + s*t'
        """
        # 1. Split.
        eps_a, eps_b = eps.chunk(2, dim=1)
        # 2. Coupling.
        # --------------------------------------------------------------------------------------------------------------
        """ The case where n_x > n_grads. """
        grads, grads_s = torch.cat([(self._grads['x_b']+self._grads['t'])*self._grads['s'], self._grads['s']], dim=1), self._grads['s']
        if len(eps) > len(grads):
            grads = grads.unsqueeze(1).expand(len(grads), len(eps)//len(grads), *grads.size()[1:]).reshape(*eps.size())
            grads_s = grads_s.unsqueeze(1).expand(len(grads_s), len(eps_b)//len(grads_s), *grads_s.size()[1:]).reshape(*eps_b.size())
        # --------------------------------------------------------------------------------------------------------------
        # (1) NN.
        output_nn = torch.cat([eps_b, eps_b], dim=1) * grads
        for module in self._nn[::-1]:
            output_nn = module.linearized_transpose(output_nn)
        # (2) Affine.
        output_a = eps_a + output_nn
        output_b = eps_b * grads_s
        # 3. Merge.
        output = torch.cat([output_a, output_b], dim=1)
        # Return
        return output

    def inverse(self, x):
        # 1. Split.
        x_a, output_b = x.chunk(2, dim=1)
        # 2. Coupling.
        # (1) NN.
        output_nn = x_a
        for module in self._nn: output_nn = module(output_nn)
        # (2) Affine.
        logs, t = output_nn.chunk(2, dim=1)
        x_b = output_b / logs.exp() - t
        # 2. Concat.
        output = torch.cat([x_a, x_b], dim=1)
        # Return
        return output


class FlowUpsample(InvertibleModule):
    """
    Flow: (c*s^2, h//s, w//s) -> (c, h, w).
        (conv + actnorm + coupling) + ... + unsqueeze.
    """
    def __init__(self, input_nc, hidden_nc, n_couplings, upsample=2):
        super(FlowUpsample, self).__init__()
        assert input_nc % 2 == 0 and input_nc % upsample**2 == 0
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Pre-conv + couplings + convs.
        self._pre_conv = InvConv2d1x1Fixed(input_nc)
        self._couplings = nn.ModuleList([AffineCoupling(input_nc, hidden_nc) for _ in range(n_couplings)])
        self._convs = nn.ModuleList([InvConv2d1x1Fixed(input_nc) for _ in range(n_couplings)])
        # 2. Unsqueeze.
        self._usqz = Unsqueeze(s=upsample)

    def forward(self, x, linearize=False, compute_logdet=False):
        """ Init logdet. """
        logdet = None
        # 1. Pre-conv + couplings + convs.
        x = self._pre_conv(x)
        for conv, coupling in zip(self._convs, self._couplings):
            # (1) Coupling.
            x = coupling(x, linearize=linearize, compute_logdet=compute_logdet)
            if compute_logdet:
                x, coupling_logdet = x
                """ Accumulate logdet. """
                logdet = coupling_logdet if logdet is None else (logdet + coupling_logdet)
            # (2) Conv.
            x = conv(x)
        # 2. Unsqueeze.
        output = self._usqz(x)
        # Return
        return (output, logdet) if compute_logdet else output

    def linearized_transpose(self, eps):
        # 1. Unsqueeze.
        output = self._usqz.linearized_transpose(eps)
        # 2. Pre-conv + couplings + convs.
        for conv, coupling in zip(self._convs[::-1], self._couplings[::-1]):
            output = conv.linearized_transpose(output)
            output = coupling.linearized_transpose(output)
        output = self._pre_conv.linearized_transpose(output)
        # Return
        return output

    def inverse(self, x):
        # 1. Unsqueeze.
        output = self._usqz.inverse(x)
        # 2. Pre-conv + couplings + convs.
        for conv, coupling in zip(self._convs[::-1], self._couplings[::-1]):
            output = conv.inverse(output)
            output = coupling.inverse(output)
        output = self._pre_conv.inverse(output)
        # Return
        return output

    ####################################################################################################################
    # Regularization.
    ####################################################################################################################

    def proxy_forward(self, e, x, sv_pred, linearize=False):
        """ Used in proxy regularization.
        :param e:
        :param x:
        :param sv_pred: The reciprocal of sv_pred.
        :param linearize:
        :return:
        """
        x = x - sv_pred + sv_pred * e
        output = self.forward(x, linearize=linearize)
        # Return
        return output

    def proxy_linearized_transpose(self, eps, sv_pred):
        """ Used in proxy regularization. """
        output = self.linearized_transpose(eps)
        output = output * sv_pred
        # Return
        return output

    def proxy_inverse(self, x, sv_pred):
        """ Used in proxy regularization. """
        output = self.inverse(x)
        output = (output - (output - sv_pred).detach()) / sv_pred
        # Return
        return output

    def compute_logsn(self, x, sv_pred, sn_power=3):
        """ Estimating the Jacobian spectral norm by using the linearized transpose. """
        # 1. Forward & LT.
        # --------------------------------------------------------------------------------------------------------------
        # (1) Forward & linearize.
        # --------------------------------------------------------------------------------------------------------------
        e = torch.ones_like(x).requires_grad_(True)
        output = self.proxy_forward(e=e, x=x, sv_pred=sv_pred, linearize=True)
        # --------------------------------------------------------------------------------------------------------------
        # (2) LT.
        # --------------------------------------------------------------------------------------------------------------
        x_t = torch.randn_like(output).requires_grad_(True)
        output_t = self.proxy_linearized_transpose(x_t, sv_pred=sv_pred)
        # 2. Compute logsn.
        logsn = fast_approx_logsn(e, output, x_t, output_t, sn_power=sn_power)
        # Return
        return logsn

    @api_empty_cache
    def eval_jacob(self, x, u_nc, log_sv_pred, jacob_size=16, **kwargs):
        """ Given x is [u;v]. """
        u_dim = u_nc * x.size(2)*x.size(3)
        ################################################################################################################
        # Compute J(x) & JTJ(x).
        ################################################################################################################
        jacob = autograd_jacob(x, func=self.forward, bsize=jacob_size)
        jtj = torch.matmul(jacob.transpose(1, 2), jacob)
        ################################################################################################################
        # Statistics.
        ################################################################################################################
        with torch.no_grad():
            # 1. Jacob orthogonality. (batch, ).
            ortho_sign = measure_ortho(jtj)
            # 2. Singular values. (batch, nc).
            svs = batch_diag(jtj).sqrt().cpu().numpy()
            svs_u, svs_v = svs[:, :u_dim], svs[:, u_dim:]
            # 3. Matching err between Jacobian s.v. & predicted s.v.
            logdet = np.array([np.linalg.slogdet(_jtj)[1] for _jtj in jtj.cpu().numpy()], dtype=np.float32)*0.5
            factor = np.exp((logdet + log_sv_pred.sum(dim=(1, 2, 3)).cpu().numpy()) / (x.size(1)*x.size(2)*x.size(3)))
            pred_svs = factor[:, np.newaxis] / log_sv_pred.reshape(x.size(0), -1).exp().cpu().numpy()
        # Get result.
        ret = {'svs_match_err': normalized_mean_absolute_err(pred=pred_svs, gt=svs)}
        if 'logdet_target' in kwargs:
            ret.update({'logdet_u0_match_err': normalized_mean_absolute_err(pred=logdet, gt=kwargs['logdet_target'].expand(logdet.shape[0]))})
        ret.update({'jacob@ortho': ortho_sign, 'svs_u': svs_u.reshape(-1, ), 'svs_v': svs_v.reshape(-1, )})
        # Return
        return ret

    @api_empty_cache
    def eval_debug(self, x, sv_pred, jacob_size=16):
        """ Given x is [u;v]. """
        ################################################################################################################
        # Compute Jacobians.
        ################################################################################################################
        """ Linearize """
        with torch.no_grad():
            y, logdet = self.forward(x, linearize=True, compute_logdet=True)
        # 1. Flow.
        flow_jacob = autograd_jacob(x, func=self.forward, bsize=jacob_size)
        flow_jacob_t = autograd_jacob(torch.randn_like(y), func=self.linearized_transpose, bsize=jacob_size)
        # 2. Composed.
        _j_param_x, _j_param_s = tuple(map(
            lambda _x: _x[:, None].expand(_x.size(0), jacob_size, *_x.size()[1:]).reshape(-1, *_x.size()[1:]), [x, sv_pred]))
        com_jacob = autograd_jacob(torch.ones_like(x), func=lambda _e: self.proxy_forward(_e, x=_j_param_x, sv_pred=_j_param_s), bsize=jacob_size)
        com_jacob_t = autograd_jacob(torch.randn_like(y), func=lambda _x: self.proxy_linearized_transpose(eps=_x, sv_pred=_j_param_s), bsize=jacob_size)
        ################################################################################################################
        # Evaluation.
        ################################################################################################################
        with torch.no_grad():
            """ LT & inv err. """
            flow_jacob_lt_err = normalized_mean_absolute_err(pred=flow_jacob_t, gt=flow_jacob.transpose(1, 2))
            flow_inv_err = normalized_mean_absolute_err(pred=self.inverse(y), gt=x)
            """ Com. """
            com_jacob_err = normalized_mean_absolute_err(pred=com_jacob, gt=flow_jacob*sv_pred.squeeze(-1).squeeze(-1).unsqueeze(1))
            com_jacob_lt_err = normalized_mean_absolute_err(pred=com_jacob_t, gt=flow_jacob_t*sv_pred.squeeze(-1).squeeze(-1).unsqueeze(2))
            """ Logdet. """
            flow_jtj = torch.matmul(flow_jacob.transpose(1, 2), flow_jacob)
            logdet_jacob = np.array([np.linalg.slogdet(_jtj)[1] for _jtj in flow_jtj.cpu().numpy()], dtype=np.float32)*0.5
            logdet_compute = logdet.total.expand(logdet_jacob.shape[0], ).cpu()
            logdet_err = normalized_mean_absolute_err(pred=logdet_compute, gt=logdet_jacob)
        # Return
        return {'flow_jacob_lt_err': flow_jacob_lt_err, 'flow_inv_err': flow_inv_err,
                'com_jacob_err': com_jacob_err, 'com_jacob_lt_err': com_jacob_lt_err, 'logdet_err': logdet_err}
