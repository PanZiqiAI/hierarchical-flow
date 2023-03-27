
import torch
from custom_pkg.pytorch.operations import BaseCriterion


# ----------------------------------------------------------------------------------------------------------------------
# Reconstruction Loss.
# ----------------------------------------------------------------------------------------------------------------------

class LossRecon(BaseCriterion):
    """ Reconstruction loss. """
    def __init__(self, mode, lmd=None):
        super(LossRecon, self).__init__(lmd=lmd)
        # Config.
        assert mode in ['l1', 'l2']
        self._mode = mode

    @staticmethod
    @torch.no_grad()
    def recon_l1err(x_real, x_recon):
        return (x_real - x_recon).abs().mean().item()

    def _call_method(self, x_real, x_recon):
        x_real, x_recon = map(lambda _x: _x.reshape(_x.size(0), -1), [x_real, x_recon])
        # Compute.
        if self._mode == 'l1': loss = (x_real - x_recon).abs().sum(1)
        else: loss = ((x_real - x_recon)**2).sum(1)
        # Return
        return loss.mean()


class LossV(BaseCriterion):
    """ Regularize training samples to be located on the generated manifold. """
    def _call_method(self, vs):
        # 1. Init results.
        loss = None
        # 2. Summarize.
        for v in vs:
            loss_cur = (v**2).mean()
            """ Accumulate. """
            loss = loss_cur if loss is None else (loss + loss_cur)
        # Return
        return loss


# ----------------------------------------------------------------------------------------------------------------------
# Jacobian Loss.
# ----------------------------------------------------------------------------------------------------------------------

class LossJacob(BaseCriterion):
    """ Jacob loss. """
    def _call_method(self, logsn, logdet_elem, **kwargs):
        # 1. Logsn = logdet = 0.
        loss_jacob = (logsn**2).mean() + (logdet_elem**2).mean()
        # 2. Logdet at (u, 0).
        if 'flow_logdet' in kwargs:
            flow_logdet, sv_predictor = kwargs['flow_logdet'], kwargs['sv_predictor']
            # Update EMA.
            sv_predictor.update_ema_u0_logdet(new=flow_logdet.detach().mean())
            # Compute loss.
            loss_jacob = loss_jacob + ((flow_logdet - sv_predictor.ema_u0_logdet)**2).mean()
        # Return
        return loss_jacob
