
import os
from tqdm import tqdm
from functools import partial
from modellib.modules import *
from utils.criterions import *
from custom_pkg.io.logger import Logger
from utils.visualize import visualize_coordinate_lines
from custom_pkg.pytorch.base_models import IterativeBaseModel
from custom_pkg.basic.metrics import TriggerLambda, FreqCounter
from custom_pkg.pytorch.operations import summarize_losses_and_backward
from custom_pkg.basic.operations import fet_d, PathPreparation, BatchSlicerInt, BatchSlicerLenObj, IterCollector


class Trainer(IterativeBaseModel):
    """ Trainer for the generator. """
    def _build_architectures(self):
        # Generator.
        generator = Generator(
            u_nc=self._cfg.args.u_nc, img_nc=self._cfg.args.img_nc, middle_u_ncs=self._cfg.args.middle_u_ncs, upsamples=self._cfg.args.upsamples,
            hidden_ncs=self._cfg.args.gen_hidden_ncs, ns_couplings=self._cfg.args.gen_ns_couplings, tracking_u_beta=self._cfg.args.tracking_u_beta)
        # SV predictors.
        sv_predictors = SVPredictorList(
            u_nc=self._cfg.args.u_nc, img_nc=self._cfg.args.img_nc, middle_u_ncs=self._cfg.args.middle_u_ncs, upsamples=self._cfg.args.upsamples,
            hidden_ncs=self._cfg.args.svp_hidden_ncs, ns_layers=self._cfg.args.svp_ns_layers)
        """ Init. """
        super(Trainer, self)._build_architectures(Gen=generator, Svp=sv_predictors)

    def _set_logs(self, **kwargs):
        super(Trainer, self)._set_logs(**kwargs)

        _log_kwargs = dict(
            formatted_prefix=self._cfg.args.desc, formatted_counters=['epoch', 'batch', 'step', 'iter'],
            append_mode=False if self._cfg.args.load_from == -1 else True, pbar=self._meters['pbar'])

        # Reconstruction.
        self._logs['log_eval_recon_cur_err'] = Logger(os.path.join(self._cfg.args.ana_train_dir, 'eval_logs'), 'recon_cur_err', **_log_kwargs)
        self._logs['log_eval_recon_from_err'] = Logger(os.path.join(self._cfg.args.ana_train_dir, 'eval_logs'), 'recon_from_err', **_log_kwargs)
        # Jacob.
        for i in range(self._Gen.n_levels):
            self._logs[f'log_eval_jacob_u0-level@{i}'] = Logger(os.path.join(self._cfg.args.ana_train_dir, 'eval_logs/jacob_u0'), f'level@{i}', **_log_kwargs)
            self._logs[f'log_eval_jacob_uv-level@{i}'] = Logger(os.path.join(self._cfg.args.ana_train_dir, 'eval_logs/jacob_uv'), f'level@{i}', **_log_kwargs)

    def _set_criterions(self):
        # Loss on V.
        self._criterions['recon'] = LossRecon(mode=self._cfg.args.recon_mode)
        self._criterions['v'] = LossV()
        # Loss on Jacobian.
        self._criterions['jacob'] = LossJacob()

    def _set_optimizers(self):
        self._optimizers['main'] = torch.optim.Adam(
            list(self._Gen.parameters())+list(self._Svp.parameters()), lr=self._cfg.args.learning_rate, betas=(0.5, 0.9))

    def _set_meters(self, **kwargs):
        super(Trainer, self)._set_meters(**kwargs)
        # --------------------------------------------------------------------------------------------------------------
        # Train.
        # --------------------------------------------------------------------------------------------------------------
        self._meters['trigger_jacob'] = TriggerLambda(
            lmd_trigger=lambda _n: self._cfg.args.freq_step_jacob != -1 and self._meters['i']['step'] % self._cfg.args.freq_step_jacob == 0,
            n_fetcher=lambda: self._meters['i']['step'])
        # --------------------------------------------------------------------------------------------------------------
        # Eval.
        # --------------------------------------------------------------------------------------------------------------
        self._meters['counter_eval_vis'] = FreqCounter(
            self._cfg.args.freq_step_eval_vis, iter_fetcher=lambda: self._meters['i']['step'])
        self._meters['counter_eval_jacob'] = FreqCounter(
            self._cfg.args.freq_step_eval_jacob, iter_fetcher=lambda: self._meters['i']['step'])

    ####################################################################################################################
    # Training.
    ####################################################################################################################

    def _deploy_batch_data(self, batch_data):
        images = batch_data.to(self._cfg.args.device)
        return images.size(0), images

    def _sampling_u(self, batch_size):
        # Reparameterize.
        """ For gauss. """
        if self._cfg.args.sampling_u_mode == 'gauss':
            u = torch.randn(batch_size, self._cfg.args.u_nc, 1, 1, device=self._cfg.args.device)
            running_mean, running_var = self._Gen.u_running_mean[None, :, None, None], self._Gen.u_running_var[None, :, None, None]
            u = u * running_var.sqrt() + running_mean
        else:
            u = torch.rand(batch_size, self._cfg.args.u_nc, 1, 1, device=self._cfg.args.device)
            running_min, running_max = self._Gen.u_running_min[None, :, None, None], self._Gen.u_running_max[None, :, None, None]
            u = u * (running_max - running_min) + running_min
        # Return
        return u

    def _sampling_vs(self, batch_size):
        # 1. Init results.
        vs = []
        # 2. Sampling for each level.
        for nc, size in self._Gen.vs_sizes:
            v = l2_normalization(torch.randn(batch_size, nc, size, size, device=self._cfg.args.device)) * self._cfg.args.sampling_v_radius
            v = v * torch.rand(batch_size, 1, 1, 1, device=self._cfg.args.device)
            """ Saving. """
            vs.append(v)
        # Return
        return vs

    def _train_step(self, packs):
        self._optimizers['main'].zero_grad()
        ################################################################################################################
        # Compute losses & update.
        ################################################################################################################
        x_manif = self._fetch_batch_data()
        u_manif, vs_manif = self._Gen.inverse(x_manif, tracking_u_stats=True)
        x_recon = None
        # --------------------------------------------------------------------------------------------------------------
        # Loss on Jacobians.
        # --------------------------------------------------------------------------------------------------------------
        losses = {}
        if self._meters['trigger_jacob'].check():
            criterion = partial(self._criterions['jacob'], lmd=self._cfg.args.lambda_jacob)
            # ----------------------------------------------------------------------------------------------------------
            # Encoded.
            # ----------------------------------------------------------------------------------------------------------
            loss_jacob_enc_u0, loss_jacob_enc_uv, x_recon = self._Gen.compute_jacob_losses(
                u=u_manif, vs=self._sampling_vs(self._cfg.args.batch_size), sv_predictors=self._Svp, criterion=criterion, sn_power=self._cfg.args.sn_power)
            # ----------------------------------------------------------------------------------------------------------
            # Random.
            # ----------------------------------------------------------------------------------------------------------
            loss_jacob_rnd_u0, loss_jacob_rnd_uv, _ = self._Gen.compute_jacob_losses(
                u=self._sampling_u(self._cfg.args.batch_size), vs=self._sampling_vs(self._cfg.args.batch_size), sv_predictors=self._Svp,
                criterion=criterion, sn_power=self._cfg.args.sn_power)
            """ Saving. """
            losses.update({
                'loss_jacob_enc_u0': loss_jacob_enc_u0, 'loss_jacob_enc_uv': loss_jacob_enc_uv,
                'loss_jacob_rnd_u0': loss_jacob_rnd_u0, 'loss_jacob_rnd_uv': loss_jacob_rnd_uv})
        # --------------------------------------------------------------------------------------------------------------
        # Losses for reconstruction.
        # --------------------------------------------------------------------------------------------------------------
        if x_recon is None: x_recon = self._Gen(u_manif)
        losses.update({
            'loss_recon': self._criterions['recon'](x_real=x_manif, x_recon=x_recon, lmd=self._cfg.args.lambda_recon),
            'loss_v': self._criterions['v'](vs_manif, lmd=self._cfg.args.lambda_v)
        })
        """ Backward. """
        summarize_losses_and_backward(*losses.values())
        ################################################################################################################
        self._optimizers['main'].step()
        """ Logging. """
        packs['log'].update({'recon_l1err': self._criterions['recon'].recon_l1err(x_manif, x_recon)})
        packs['log'].update(fet_d(losses, 'loss_recon', 'loss_v', lmd_v=lambda _v: _v.item()))
        packs['log'].update(fet_d(losses, prefix='loss_jacob_', lmd_v=lambda _v: _v.item()/self._Gen.n_levels))

    def _process_log(self, packs, **kwargs):

        def _lmd_generate_log():
            packs['tfboard'].update({
                'train/recon_l1err': fet_d(packs['log'], 'recon_l1err'),
                'train/losses': fet_d(packs['log'], prefix='loss_', replace='')
            })

        super(Trainer, self)._process_log(packs, lmd_generate_log=_lmd_generate_log)

    def _process_after_step(self, packs, **kwargs):
        # 1. Logging.
        self._process_log(packs, **kwargs)
        # 2. Evaluation.
        if self._meters['counter_eval_vis'].check():
            if self._cfg.args.dataset == 'circle': self._eval_vis_1d_manif_in_2d()
        if self._meters['counter_eval_jacob'].check():
            self._eval_jacob()
        """ lr & chkpt. """
        self._process_chkpt_and_lr()

    ####################################################################################################################
    # Evaluation.
    ####################################################################################################################

    @torch.no_grad()
    def _eval_vis_1d_manif_in_2d(self):
        assert self._cfg.args.u_nc == 1 and self._cfg.args.img_nc == 2 and self._cfg.args.middle_u_ncs == [] and self._cfg.args.upsamples == [1]
        ################################################################################################################
        # Visualize coordinate lines.
        ################################################################################################################
        # Determine u_min & u_max.
        us = []
        for x in BatchSlicerLenObj(self._data['train'].dataset, batch_size=self._cfg.args.batch_size):
            x = torch.from_numpy(x).to(self._cfg.args.device)
            us.append(self._Gen.inverse(x)[0].cpu().numpy())
        us = np.concatenate(us)
        u1, u2 = us.min(0).item(), us.max(0).item()
        """ Visualize. """
        with PathPreparation(self._cfg.args.eval_vis_dir, 'flow_coordinates') as save_dir:
            visualize_coordinate_lines(
                u1, u2, v_radius=self._cfg.args.sampling_v_radius, n_grids=self._cfg.args.eval_vis_n_grids,
                n_samples_curve=self._cfg.args.eval_vis_n_samples_curve, func_flow=self._Gen.flows[0], device=self._cfg.args.device,
                save_path=os.path.join(save_dir, f"step[{self._meters['i']['step']}].png"))
    
    @api_empty_cache
    def _eval_jacob(self):
        ################################################################################################################
        # Get evaluation results.
        ################################################################################################################
        # 1. Init results.
        results_u0, results_uv = [IterCollector() for _ in range(self._Gen.n_levels)], [IterCollector() for _ in range(self._Gen.n_levels)]
        # 2. Collect from batch.
        pbar = tqdm(total=self._cfg.args.eval_jacob_n_samples, desc="Evaluating flow's Jacobian")
        for batch_size in BatchSlicerInt(self._cfg.args.eval_jacob_n_samples, self._cfg.args.eval_jacob_batch_size):
            batch_ret_u0_levels, batch_ret_uv_levels = self._Gen.eval_jacob(
                u=self._sampling_u(batch_size), vs=self._sampling_vs(batch_size), sv_predictors=self._Svp,
                jacob_size=self._cfg.args.eval_jacob_ag_bsize)
            """ Saving for each level. """
            for collector, batch_ret in zip(results_u0, batch_ret_u0_levels): collector.collect(batch_ret)
            for collector, batch_ret in zip(results_uv, batch_ret_uv_levels): collector.collect(batch_ret)
            """ Progress. """
            pbar.update(batch_size)
        pbar.close()
        # 3. Get results.
        results_u0, results_uv = map(lambda _r: [_c.pack() for _c in _r], [results_u0, results_uv])
        """ Compute svs avg. & std. """

        def _process_dict(_ret):
            _svs_u = _ret.pop('svs_u')
            _svs_v = _ret.pop('svs_v')
            _ret = {_k: _v.mean().item() for _k, _v in _ret.items()}
            _ret.update({
                'jacob_u@normal@avg': _svs_u.mean().item(), 'jacob_u@normal@std': _svs_u.std(ddof=1).item(),
                'jacob_v@normal@avg': _svs_v.mean().item(), 'jacob_v@normal@std': _svs_v.std(ddof=1).item()})
            return _ret

        results_u0 = [_process_dict(_ret) for _ret in results_u0]
        results_uv = [_process_dict(_ret) for _ret in results_uv]
        ################################################################################################################
        # Logging.
        ################################################################################################################
        """ Logger. """
        for i, (ret_u0, ret_uv) in enumerate(zip(results_u0, results_uv)):
            self._logs[f'log_eval_jacob_u0-level@{i}'].info_formatted(counters=self._meters['i'], items=ret_u0)
            self._logs[f'log_eval_jacob_uv-level@{i}'].info_formatted(counters=self._meters['i'], items=ret_uv)
        """ Tfboard. """
        for i, (ret_u0, ret_uv) in enumerate(zip(results_u0, results_uv)):
            self._logs['tfboard'].add_multi_scalars(multi_scalars={
                f'eval/jacob_u0/{k}': {'': v} for k, v in ret_u0.items()}, global_step=self._meters['i']['step'])
            self._logs['tfboard'].add_multi_scalars(multi_scalars={
                f'eval/jacob_uv/{k}': {'': v} for k, v in ret_uv.items()}, global_step=self._meters['i']['step'])

    @api_empty_cache
    def eval_debug(self, n_samples):
        ################################################################################################################
        # Get evaluation results.
        ################################################################################################################
        # 1. Init results.
        results = [IterCollector() for _ in range(self._Gen.n_levels)]
        # 2. Collect from batch.
        pbar = tqdm(total=n_samples, desc="Debuging")
        for batch_size in BatchSlicerInt(n_samples, self._cfg.args.eval_jacob_batch_size):
            batch_ret_levels = self._Gen.eval_debug(
                u=self._sampling_u(batch_size), vs=self._sampling_vs(batch_size), sv_predictors=self._Svp,
                jacob_size=self._cfg.args.eval_jacob_ag_bsize)
            """ Saving for each level. """
            for collector, batch_ret in zip(results, batch_ret_levels): collector.collect(batch_ret)
            """ Progress. """
            pbar.update(batch_size)
        pbar.close()
        # 3. Get results.
        results = [_c.pack(reduction=lambda _x: np.mean(_x).item()) for _c in results]
        ################################################################################################################
        # Logging.
        ################################################################################################################
        for i, ret in enumerate(results):
            print('#'*100)
            print(f'Level@{i}')
            print('#'*100)
            print(ret)
