
from modellib.layers import *


########################################################################################################################
# Proxy sv. predictors.
########################################################################################################################

class SVPredictor(nn.Module):
    """ Proxy singular values predictor. """
    def __init__(self, x_nc, u_nc, hidden_nc, n_layers):
        super(SVPredictor, self).__init__()
        # Config.
        self._u_nc, self._v_nc = u_nc, x_nc-u_nc
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        module = []
        for index in range(n_layers):
            module.extend([
                nn.Conv2d(x_nc, hidden_nc, kernel_size=1), nn.ReLU(inplace=True),
                nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(hidden_nc, u_nc if index == n_layers-1 else x_nc, kernel_size=1)])
            if index < n_layers-1: module.extend([nn.ReLU(inplace=True)])
        # --------------------------------------------------------------------------------------------------------------
        # Global logs for u0.
        # --------------------------------------------------------------------------------------------------------------
        """ For u0, all u-dimensions share the same s.v. due to the isometry property on the manifold. """
        self.register_buffer("_ema_u0_logdet", torch.zeros(1))
        """ Setup. """
        self._module = nn.Sequential(*module)

    @property
    def ema_u0_logdet(self):
        return self._ema_u0_logdet

    def update_ema_u0_logdet(self, new):
        self._ema_u0_logdet = update_ema(self._ema_u0_logdet, new)

    def forward(self, x, mode, **kwargs):
        """ Given x (batch, nc, h, w),
            - the singular values for u is predicted using self._module, the shape is (batch, u_nc, h, w).
            - the singular values for v is always ones, the shape is (batch, v_nc, h, w).

            For J_com = J_f * diag(w1, w2, ..., wK) satisfying that all singular values are equal, we have that
        sigma_j = sqrt[K]{d} / w_j, where d is the determinant of J_com. To regularize sigma_j=1 for V, we know that
        sqrt[K]{d} / w_j = 1 should be satisfied for V, namely w_j = sqrt[K]{d} for V. Therefore, we have that
            prod_j=1^K sigma_j*w_j = d                  =====>
            detJ * (prod_U w_j) * (prod_V w_j) = d      =====>
            detJ * (prod_U w_j) * d^{V/K} = d           =====>
            detJ * (prod_U w_j) = d^(U/K)               =====>
            w_j (for V) = d^(1/K) = {detJ * (prod_U w_j)}^(1/U) = sqrt[U]{ detJ * (prod_U w_j) }
        """
        assert mode in ['u0', 'uv']
        # --------------------------------------------------------------------------------------------------------------
        # Forward for U.
        # --------------------------------------------------------------------------------------------------------------
        if mode == 'u0':
            flow_logdet = self._ema_u0_logdet.expand(x.size(0))
            logs_u = ((-self._ema_u0_logdet) / (self._u_nc*x.size(2)*x.size(3))).expand(x.size(0), self._u_nc, *x.size()[2:])
        else:
            flow_logdet = kwargs['flow_logdet'].total
            logs_u = self._module(x)
        # --------------------------------------------------------------------------------------------------------------
        # Compute for V.
        # --------------------------------------------------------------------------------------------------------------
        logs_v = (flow_logdet + logs_u.sum(dim=(1, 2, 3))) / (logs_u.size(1)*logs_u.size(2)*logs_u.size(3))
        logs_v = logs_v[:, None, None, None].expand(x.size(0), self._v_nc, *x.size()[2:])
        # --------------------------------------------------------------------------------------------------------------
        # Setup.
        # --------------------------------------------------------------------------------------------------------------
        logs = torch.cat([logs_u, logs_v], dim=1)
        # Return
        return logs


class SVPredictorList(nn.ModuleList):
    """ List of singular values predictors, corresponding to flow modules. """
    def __init__(self, u_nc, img_nc, middle_u_ncs, upsamples, hidden_ncs, ns_layers):
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        modules = []
        for u_nc1, u_nc2, s2, h_nc, n_layers in zip([u_nc]+middle_u_ncs, middle_u_ncs+[img_nc], upsamples, hidden_ncs, ns_layers):
            modules.append(SVPredictor(x_nc=u_nc2*s2*s2, u_nc=u_nc1, hidden_nc=h_nc, n_layers=n_layers))
        """ Setup. """
        super(SVPredictorList, self).__init__(modules)


########################################################################################################################
# Generator.
########################################################################################################################

class Generator(InvertibleModule):
    """ Generator. For example, parameters
        - middle_u_ncs (L-1):   u_nc1, u_nc2, ..., u_nc(L-1).
        - upsamples (L):        s1, s2, ..., sL.
        - hidden_ncs (L):
        - n_couplings (L):
    corresponds to hierarchical features (u) of
        (u_nc, 1, 1) < (u_nc1, s1, s1) < (u_nc2, s1*s2, s1*s2) < ... < (img_nc, s1*s2*...*sL, s1*s2*...*sL),
    where s1*s2*...*sL=img_size.
    """
    def __init__(self, u_nc, img_nc, middle_u_ncs, upsamples, hidden_ncs, coupling_mode, ns_couplings):
        super(Generator, self).__init__()
        # Config.
        self._upsamples = upsamples
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        self._paddings, self._flows = nn.ModuleList([]), nn.ModuleList([])
        for u_nc1, u_nc2, s2, h_nc, n_coup in zip([u_nc]+middle_u_ncs, middle_u_ncs+[img_nc], upsamples, hidden_ncs, ns_couplings):
            x_nc = u_nc2*s2**2
            # 1. Padding.
            self._paddings.append(Padding(x_nc=x_nc, u_nc=u_nc1))
            # 2. Flow.
            self._flows.append(FlowUpsample(input_nc=x_nc, hidden_nc=h_nc, coupling_mode=coupling_mode, n_couplings=n_coup, upsample=s2))
        # --------------------------------------------------------------------------------------------------------------
        # Tracking u stats.
        # --------------------------------------------------------------------------------------------------------------
        # For gauss.
        self.register_buffer("_u_running_mean", torch.zeros(u_nc))
        self.register_buffer("_u_running_var", torch.ones(u_nc))
        # For uni.
        self.register_buffer("_u_running_min", torch.zeros(u_nc))
        self.register_buffer("_u_running_max", torch.zeros(u_nc))

    @property
    def n_levels(self):
        return len(self._paddings)

    def vs_sizes(self, img_size):
        # 1. Init results.
        results, size = [], img_size
        # 2. Compute each level.
        for padding, upsample in zip(self._paddings[::-1], self._upsamples[::-1]):
            """ Update size. """
            size = size//upsample
            """ Saving. """
            results.append((padding.v_nc, size))
        # Return
        return results[::-1]

    @property
    def flows(self):
        return self._flows

    @property
    def u_running_min(self):
        return self._u_running_min

    @property
    def u_running_max(self):
        return self._u_running_max

    @property
    def u_running_mean(self):
        return self._u_running_mean

    @property
    def u_running_var(self):
        return self._u_running_var

    # ------------------------------------------------------------------------------------------------------------------
    # Computation.
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, u, compute_logdet=False):
        flows_logdet = []
        # Forward through each flow module.
        for padding, flow in zip(self._paddings, self._flows):
            u = flow(padding(u), compute_logdet=compute_logdet)
            if compute_logdet:
                u, flow_logdet = u
                """ Accumulate logdet. """
                flows_logdet.append(flow_logdet)
        # Return
        return (u, flows_logdet) if compute_logdet else u

    def inverse(self, x, tracking_u_stats=False):
        # 1. Init results.
        u, vs = x, []
        # 2. Inverse.
        for padding, flow in zip(self._paddings[::-1], self._flows[::-1]):
            u, v = padding.inverse(flow.inverse(u))
            """ Saving. """
            vs.append(v)
        """ Tracking u stats. """
        if tracking_u_stats:
            # For gauss.
            self._u_running_mean = update_ema(self._u_running_mean, u.detach().mean(dim=(0, 2, 3)))
            self._u_running_var = update_ema(self._u_running_var, u.detach().var(dim=(0, 2, 3)))
            # For uni.
            batch_min, batch_max = map(lambda _func: _func(
                u.detach().transpose(0, 1).reshape(u.size(1), -1), dim=1).values, [torch.min, torch.max])
            self._u_running_min = update_ema(self._u_running_min, batch_min)
            self._u_running_max = update_ema(self._u_running_max, batch_max)
        # Return
        return u, vs

    def compute_jacob_losses(self, u, vs, sv_predictors, criterion, sn_power=3):
        """ Compute Jacob losses for each level. """
        # 1. Init results.
        losses_u0, losses_uv, packs_u0_mean, packs_u0_std, packs_uv_mean, packs_uv_std, = [], [], [], [], [], []
        # 2. Compute each level.
        for padding, flow, v, sv_pred in zip(self._paddings, self._flows, vs, sv_predictors):

            def _compute_jacob_loss(_v):
                _x = padding(u, _v)
                _out, _flow_logdet = flow(_x, compute_logdet=True)
                _log_sv_pred = sv_pred(_x, **(dict(mode='u0') if _v is None else dict(mode='uv', flow_logdet=_flow_logdet)))
                """ Compute logsn & logdet. """
                _com_logsn = flow.compute_logsn(_x, sv_pred=_log_sv_pred.exp(), sn_power=sn_power)
                _flow_logdet_elem, _svp_logdet_elem = _flow_logdet.elem, _log_sv_pred.mean(dim=(1, 2, 3))
                _com_logdet_elem = _flow_logdet_elem + _svp_logdet_elem
                """ Compute loss & backward. """
                _losses_cur = criterion(
                    _com_logsn, _com_logdet_elem, **(dict(flow_logdet=_flow_logdet.total, sv_predictor=sv_pred) if _v is None else dict()))
                # Return
                _packs_mean, _packs_std = {}, {}
                for _k in ['com_logsn', 'com_logdet_elem', 'flow_logdet_elem', 'svp_logdet_elem']:
                    _packs_mean[_k] = eval(f"_{_k}").mean().item()
                    _packs_std[_k] = eval(f"_{_k}").std(unbiased=False).item()
                if _v is None:
                    _packs_mean.update({'flow_logdet_total@u0': _flow_logdet.total.mean().item(), 'flow_logdet_ema@u0': sv_pred.ema_u0_logdet.item()})
                    _packs_std.update({'flow_logdet_total@u0': _flow_logdet.total.std(unbiased=False).item(), 'flow_logdet_ema@u0': None})
                return _losses_cur, _packs_mean, _packs_std, _out

            # ----------------------------------------------------------------------------------------------------------
            # 1. For (u, 0)
            losses_u0_cur, packs_u0_mean_cur, packs_u0_std_cur, next_u = _compute_jacob_loss(_v=None)
            """ Accumulate. """
            losses_u0.append(losses_u0_cur)
            packs_u0_mean.append(packs_u0_mean_cur)
            packs_u0_std.append(packs_u0_std_cur)
            # 2. For (u, v)
            losses_uv_cur, packs_uv_mean_cur, packs_uv_std_cur, _ = _compute_jacob_loss(_v=v)
            """ Accumulate. """
            losses_uv.append(losses_uv_cur)
            packs_uv_mean.append(packs_uv_mean_cur)
            packs_uv_std.append(packs_uv_std_cur)
            # ----------------------------------------------------------------------------------------------------------
            """ Move to next level. """
            u = next_u
        # Return
        return losses_u0, losses_uv, packs_u0_mean, packs_u0_std, packs_uv_mean, packs_uv_std, u

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluation.
    # ------------------------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def eval_recon(self, x):
        """ Return: List of (cur_level_input, cur_level_recon, fomr_level_recon). Levels order is L, L-1, ..., 1. """
        # 1. Init results.
        results = []
        # 2. Compute each level.
        cur_input = x
        for level in list(range(len(self._flows)))[::-1]:
            u, _ = self._paddings[level].inverse(self._flows[level].inverse(cur_input))
            # (1) Recon for current level.
            cur_recon = self._flows[level](self._paddings[level](u))
            # (2) Recon from current level.
            from_recon = cur_recon
            for padding, flow in zip(self._paddings[level+1:], self._flows[level+1:]): from_recon = flow(padding(from_recon))
            """ Saving. """
            results.append((cur_input.cpu(), cur_recon.cpu(), from_recon.cpu()))
            """ Update. """
            cur_input = u
        # Return
        return results[::-1]

    @api_empty_cache
    def eval_jacob(self, u, vs, sv_predictors, jacob_size):
        """ Evaluate Jacob for each flow module. """
        # 1. Init results.
        ret_u0_levels, ret_uv_levels = [], []
        # 2. Compute from layer 1 to layer L.
        for padding, flow, v, sv_pred in zip(self._paddings, self._flows, vs, sv_predictors):
            # ----------------------------------------------------------------------------------------------------------
            # For u0.
            # ----------------------------------------------------------------------------------------------------------
            x = padding(u)
            ret_u0_levels.append(flow.eval_jacob(
                x=x, u_nc=padding.u_nc, log_sv_pred=sv_pred(x, mode='u0'), jacob_size=jacob_size, logdet_target=sv_pred.ema_u0_logdet))
            # ----------------------------------------------------------------------------------------------------------
            # For uv.
            # ----------------------------------------------------------------------------------------------------------
            x = padding(u, v)
            _, logdet = flow(x, compute_logdet=True)
            ret_uv_levels.append(flow.eval_jacob(
                x=x, u_nc=padding.u_nc, log_sv_pred=sv_pred(x, mode='uv', flow_logdet=logdet), jacob_size=jacob_size))
            """ Move to next u. """
            u = flow(padding(u))
        # Return
        return ret_u0_levels, ret_uv_levels

    @api_empty_cache
    def eval_debug(self, u, vs, sv_predictors, jacob_size):
        """ Debug for each flow module. """
        # 1. Init results.
        ret_levels = []
        # 2. Compute from layer 1 to layer L.
        for padding, flow, v, sv_pred in zip(self._paddings, self._flows, vs, sv_predictors):
            x = padding(u, v)
            _, logdet = flow(x, compute_logdet=True)
            ret_levels.append(flow.eval_debug(
                x=x, sv_pred=sv_pred(x, mode='uv', flow_logdet=logdet).exp(), jacob_size=jacob_size))
            """ Move to next u. """
            u = flow(padding(u))
        # Return
        return ret_levels
