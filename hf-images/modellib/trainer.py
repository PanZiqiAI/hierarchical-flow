
import os
from tqdm import tqdm
from modellib.modules import *
from utils.criterions import *
from custom_pkg.io.logger import Logger
from torchvision.utils import save_image
from custom_pkg.pytorch.base_models import IterativeBaseModel
from custom_pkg.basic.metrics import TriggerLambda, FreqCounter
from custom_pkg.pytorch.operations import summarize_losses_and_backward
from utils.visualize import visualize_coordinate_lines, generate_train_jacob_log
from custom_pkg.basic.operations import fet_d, PathPreparation, BatchSlicerInt, BatchSlicerLenObj, IterCollector


class Trainer(IterativeBaseModel):
    """ Trainer for the generator. """
    def _build_architectures(self):
        # Generator.
        generator = Generator(
            u_nc=self._cfg.args.u_nc, img_nc=self._cfg.args.img_nc, middle_u_ncs=self._cfg.args.middle_u_ncs, upsamples=self._cfg.args.upsamples,
            hidden_ncs=self._cfg.args.gen_hidden_ncs, coupling_mode="affine" if self._cfg.args.freq_step_jacob > 0 else "additive", ns_couplings=self._cfg.args.gen_ns_couplings)
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

        # Train@Jacob.
        self._logs['log_train_jacob'] = Logger(self._cfg.args.ana_train_dir, 'train_jacob', **_log_kwargs)

        # Reconstruction.
        self._logs['log_eval_recon_cur_err'] = Logger(os.path.join(self._cfg.args.ana_train_dir, 'eval_logs'), 'recon_cur_err', **_log_kwargs)
        self._logs['log_eval_recon_from_err'] = Logger(os.path.join(self._cfg.args.ana_train_dir, 'eval_logs'), 'recon_from_err', **_log_kwargs)
        # Jacob.
        for i in range(self._Gen.n_levels):
            self._logs[f'log_eval_jacob_u0-level@{i}'] = Logger(os.path.join(self._cfg.args.ana_train_dir, 'eval_logs/jacob_u0'), f'level@{i}', **_log_kwargs)
            self._logs[f'log_eval_jacob_uv-level@{i}'] = Logger(os.path.join(self._cfg.args.ana_train_dir, 'eval_logs/jacob_uv'), f'level@{i}', **_log_kwargs)

    def _set_criterions(self):
        # Loss on V.
        self._criterions['recon'] = LossRecon(mode=self._cfg.args.recon_mode, lmd=self._cfg.args.lambda_recon)
        self._criterions['v'] = LossV(lmd=self._cfg.args.lambda_v)
        # Loss on Jacobian.
        self._criterions['jacob'] = LossJacob(lmd=self._cfg.args.lambda_jacob)

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
        for nc, size in self._Gen.vs_sizes(self._cfg.args.img_size):
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
        x = self._fetch_batch_data()
        u, vs = self._Gen.inverse(x, tracking_u_stats=True)
        x_recon = None
        # --------------------------------------------------------------------------------------------------------------
        # Loss on Jacobians.
        # --------------------------------------------------------------------------------------------------------------
        losses = {}
        if self._meters['trigger_jacob'].check():
            # ----------------------------------------------------------------------------------------------------------
            # Encoded.
            # ----------------------------------------------------------------------------------------------------------
            losses_enc_u0, losses_enc_uv, packs_enc_u0_mean, packs_enc_u0_std, packs_enc_uv_mean, packs_enc_uv_std, x_recon = \
                self._Gen.compute_jacob_losses(u=u, vs=self._sampling_vs(self._cfg.args.batch_size), sv_predictors=self._Svp,
                                               criterion=self._criterions['jacob'], sn_power=self._cfg.args.sn_power)
            # ----------------------------------------------------------------------------------------------------------
            # Random.
            # ----------------------------------------------------------------------------------------------------------
            losses_rnd_u0, losses_rnd_uv, packs_rnd_u0_mean, packs_rnd_u0_std, packs_rnd_uv_mean, packs_rnd_uv_std, _ = \
                self._Gen.compute_jacob_losses(u=self._sampling_u(self._cfg.args.batch_size), vs=self._sampling_vs(self._cfg.args.batch_size), sv_predictors=self._Svp,
                                               criterion=self._criterions['jacob'], sn_power=self._cfg.args.sn_power)
            """ Saving. """
            for name in ['enc_u0', 'enc_uv', 'rnd_u0', 'rnd_uv']:
                # Loss of levels.
                for level, loss_value in enumerate(eval(f"losses_{name}")):
                    losses.update({f"loss_level{level}_{name}": loss_value})
                # Pack of levels.
                packs['jacob_mean'].update({name: eval(f"packs_{name}_mean")})
                packs['jacob_std'].update({name: eval(f"packs_{name}_std")})
        # --------------------------------------------------------------------------------------------------------------
        # Losses on Reconstruction.
        # --------------------------------------------------------------------------------------------------------------
        if x_recon is None: x_recon = self._Gen(u)
        losses.update({'loss_recon': self._criterions['recon'](x_real=x, x_recon=x_recon), 'loss_v': self._criterions['v'](vs)})
        """ Backward. """
        summarize_losses_and_backward(*losses.values())
        ################################################################################################################
        self._optimizers['main'].step()
        """ Logging. """
        packs['log'].update({'recon_l1err': self._criterions['recon'].recon_l1err(x, x_recon)})
        packs['log'].update(fet_d(losses, 'loss_recon', 'loss_v', lmd_v=lambda _v: _v.item()))
        packs['log'].update(fet_d(losses, prefix='loss_level_', lmd_v=lambda _v: _v.item()))

    def _process_log(self, packs, **kwargs):

        def _lmd_generate_log():
            packs['tfboard'].update({
                'train/recon_l1err': fet_d(packs['log'], 'recon_l1err'),
                'train/losses': fet_d(packs['log'], prefix='loss_', replace='')
            })

        def _lmd_process_log(_log_pack):
            return {'items': _log_pack, 'no_display_keys': list(filter(lambda _x: _x.startswith('loss_level_'), _log_pack.keys()))}

        super(Trainer, self)._process_log(packs, lmd_generate_log=_lmd_generate_log, lmd_process_log=_lmd_process_log, **kwargs)
        # Write train Jacob.
        if self._meters['counter_log'].status and (packs['jacob_mean'] and packs['jacob_std']):
            self._logs['log_train_jacob'].info_individually(
                f"step[{self._meters['i']['step']}] - mean&std\n" + generate_train_jacob_log(packs['jacob_mean']) + generate_train_jacob_log(packs['jacob_std']))

    def _process_after_step(self, packs, **kwargs):
        """ Saving packs of current step. """
        self._meters['tmp-step_packs'] = packs
        # 1. Logging.
        self._process_log(packs, **kwargs)
        # 2. Evaluation.
        if self._meters['counter_eval_vis'].check():
            self._eval_recon()
        if self._meters['counter_eval_jacob'].check():
            self._eval_jacob()
        """ lr & chkpt. """
        self._process_chkpt_and_lr()

    def _init_packs(self, *args, **kwargs):
        return super()._init_packs('jacob_mean', 'jacob_std')

    ####################################################################################################################
    # Evaluation.
    ####################################################################################################################

    @torch.no_grad()
    def _eval_recon(self):
        # --------------------------------------------------------------------------------------------------------------
        # Get evaluation results.
        # --------------------------------------------------------------------------------------------------------------
        n_samples = self._cfg.args.eval_recon_n_samples
        # 1. Init results.
        results_cur_err, results_from_err, results_x_real, results_from_recon, counts = [], [], [], [], 0
        # 2. Compute.
        for batch_data in self._data['eval']:
            batch_data = batch_data.to(self._cfg.args.device)
            cur_input, cur_recon, from_recon = list(zip(*self._Gen.eval_recon(batch_data)))
            """ Compute recon err. """
            cur_err = [(_x-_x_recon).abs().mean(dim=(1, 2, 3)).cpu() for _x, _x_recon in zip(cur_input, cur_recon)]
            from_err = [(batch_data.cpu()-_x_recon).abs().mean(dim=(1, 2, 3)) for _x_recon in from_recon]
            """ Saving. """
            results_cur_err.append(cur_err)
            results_from_err.append(from_err)
            results_x_real.append(batch_data.cpu())
            results_from_recon.append(torch.cat([_x[:, None] for _x in from_recon], dim=1))     # (batch, level, c, h, w).
            counts += len(batch_data)
            if counts >= n_samples: break
        # 3. Get results.
        results_cur_err = {f'level@{i}': v for i, v in enumerate(list(
            map(lambda _x: torch.cat(_x)[:n_samples].mean().item(), list(zip(*results_cur_err)))))}
        results_from_err = {f'level@{i}': v for i, v in enumerate(list(
            map(lambda _x: torch.cat(_x)[:n_samples].mean().item(), list(zip(*results_from_err)))))}
        results_x_real, results_from_recon = map(lambda _x: torch.cat(_x)[:n_samples], [results_x_real, results_from_recon])
        # --------------------------------------------------------------------------------------------------------------
        # Logging errs.
        # --------------------------------------------------------------------------------------------------------------
        """ Logger. """
        self._logs['log_eval_recon_cur_err'].info_formatted(counters=self._meters['i'], items=results_cur_err)
        self._logs['log_eval_recon_from_err'].info_formatted(counters=self._meters['i'], items=results_from_err)
        """ Tfboard. """
        self._logs['tfboard'].add_multi_scalars(multi_scalars={
            'eval/recon/cur': results_cur_err, 'eval/recon/from': results_from_err
        }, global_step=self._meters['i']['step'])
        # --------------------------------------------------------------------------------------------------------------
        # Visualize from reconstructions.
        # --------------------------------------------------------------------------------------------------------------
        x = torch.cat([results_x_real.unsqueeze(1), results_from_recon], dim=1)
        x = x.reshape(-1, *x.size()[2:])*0.5 + 0.5
        """ Saving. """
        with PathPreparation(self._cfg.args.eval_vis_dir, 'recon') as save_dir:
            save_image(x, os.path.join(save_dir, f"step[{self._meters['i']['step']}].png"))

    @api_empty_cache
    def _eval_jacob_method(self):
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
        # Return
        return results_u0, results_uv

    def _eval_jacob(self):
        results_u0, results_uv = self._eval_jacob_method()
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

    ####################################################################################################################
    # Post evaluation
    ####################################################################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # Distance preserving.
    # ------------------------------------------------------------------------------------------------------------------

    def eval_dist_pres(self):
        # 1. Jacob.
        results = self._eval_dist_pres_jacob()
        # 2. Circle.
        if self._cfg.args.dataset == 'circle':
            results.update({'marco': self._eval_dist_pres_circle()})
        # Return
        print(results)

    @torch.no_grad()
    def _eval_dist_pres_circle(self):
        """ Evaluating the distance preserving property for the circle manifold. """
        # Compute results.
        results_manif, results_emb = [], []
        for x in BatchSlicerLenObj(self._data['eval'].dataset, batch_size=self._cfg.args.batch_size):
            x1 = torch.from_numpy(x).to(self._cfg.args.device)
            x2 = x1[torch.randperm(len(x1), device=x1.device)]
            # Compute ground-truth manifold distance. (batch, )
            manif_dist = self._data['eval'].dataset.manifold_distance(x1, x2).cpu()
            # Compute predicted manifold distance. (batch, )
            (u1, _), (u2, _) = self._Gen.inverse(x1), self._Gen.inverse(x2)
            emb_dist = ((u1 - u2)**2).sum(dim=1).sqrt().cpu()
            """ Saving. """
            results_manif.append(manif_dist)
            results_emb.append(emb_dist)
        results_manif, results_emb = map(lambda _x: torch.cat(_x), [results_manif, results_emb])
        # Measure STD of manif_dist/emb_dist.
        ratios = results_manif / results_emb
        metric = (torch.std(ratios, unbiased=False) / torch.mean(ratios)).item()
        # Return
        return metric

    @api_empty_cache
    def _eval_dist_pres_jacob(self):
        """ Evaluating the distance preserving property based on the generator Jacobian. """
        # 1. Init results.
        results_ortho_sign, results_svs = [], []
        # 2. Collect from batch.
        pbar = tqdm(total=self._cfg.args.eval_dist_pres_n_samples, desc="Evaluating distance preserving property based on Jacobian")
        for batch_size in BatchSlicerInt(self._cfg.args.eval_dist_pres_n_samples, self._cfg.args.eval_dist_pres_batch_size):
            jacob = autograd_jacob(self._sampling_u(batch_size), func=self._Gen.forward, bsize=self._cfg.args.eval_dist_pres_ag_bsize)
            jtj = torch.matmul(jacob.transpose(1, 2), jacob)
            # Measure orthogonality. (batch, )
            ortho_sign = torch.from_numpy(measure_ortho(jtj))
            # Measure SVS. (batch, nc) -> (batch*nc, )
            svs = batch_diag(jtj).sqrt().reshape(-1, ).cpu()
            """ Saving. """
            results_ortho_sign.append(ortho_sign)
            results_svs.append(svs)
            """ Progress. """
            pbar.update(batch_size)
        pbar.close()
        # 3. Get results.
        metric_ortho_sign = torch.cat(results_ortho_sign).mean().item()
        results_svs = torch.cat(results_svs)
        metric_svs_cv = (torch.std(results_svs, unbiased=False) / torch.mean(results_svs)).item()
        # Return
        return {'jacob_ortho': metric_ortho_sign, 'jacob_svs': metric_svs_cv}

    # ------------------------------------------------------------------------------------------------------------------
    # Rigorous projection.
    # ------------------------------------------------------------------------------------------------------------------

    def eval_rig_proj(self):
        ################################################################################################################
        # Jacob.
        ################################################################################################################
        results_u0, results_uv = self._eval_jacob_method()
        # --------------------------------------------------------------------------------------------------------------
        # Show u0.
        # --------------------------------------------------------------------------------------------------------------
        print('#'*100)
        print(f"(u,0)")
        print('#'*100)
        for level, ret in enumerate(results_u0): print("level[%d]: %s" % (level, ret))
        # --------------------------------------------------------------------------------------------------------------
        # Show uv.
        # --------------------------------------------------------------------------------------------------------------
        print('#'*100)
        print(f"(u,v)")
        print('#'*100)
        for level, ret in enumerate(results_uv): print("level[%d]: %s" % (level, ret))
        ################################################################################################################
        # Circle.
        ################################################################################################################
        if self._cfg.args.dataset == 'circle':
            results = self._eval_rig_proj_circle()
            print('#'*100)
            print(f"Marco metric: {results}")

    @torch.no_grad()
    def _eval_rig_proj_circle(self):
        """ Evaluting the rigorous projection property for the circle manifold. """
        # Compute results.
        results_gt, results_pred = [], []
        for batch_size in BatchSlicerInt(self._cfg.args.eval_rig_proj_n_samples, self._cfg.args.eval_rig_proj_batch_size):
            v = self._sampling_vs(batch_size)[0].squeeze(-1).squeeze(-1).squeeze(-1)
            samples = self._data['eval'].dataset.ambient_sampling(v=v)
            # Compute ground-truth projection.
            proj_gt = self._data['eval'].dataset.projection(samples)
            # Compute predicted projection.
            proj_pred = self._Gen(self._Gen.inverse(samples)[0])
            """ Saving. """
            results_gt.append(proj_gt)
            results_pred.append(proj_pred)
        results_gt, results_pred = map(lambda _x: torch.cat(_x), [results_gt, results_pred])
        # Measure distance.
        metric = ((results_gt - results_pred)**2).sum(dim=(1, 2, 3)).sqrt().mean().item()
        # Return
        return metric

    # ------------------------------------------------------------------------------------------------------------------
    # Out-of-distribution detection.
    # ------------------------------------------------------------------------------------------------------------------

    def eval_ood(self):
        if self._cfg.args.dataset == 'circle':
            metric = self._eval_ood_circle()
        else:
            metric = self._eval_ood_eps()
        # Log.
        print("TPR95: %f." % metric)

    @torch.no_grad()
    def _eval_ood_circle(self):
        # Predict distance to manifold. (n_samples, )
        in_distances, out_distances = [], []
        for batch_size in BatchSlicerInt(self._cfg.args.eval_ood_n_samples, self._cfg.args.eval_ood_batch_size):
            # Get in-samples.
            v_in = torch.rand(batch_size, device=self._cfg.args.device) * self._cfg.args.eval_ood_out_circle_thr
            in_samples = self._data['eval'].dataset.ambient_sampling(v=v_in)
            # Get out-samples.
            v_out = torch.rand(batch_size, device=self._cfg.args.device) * (
                    self._cfg.args.sampling_v_radius - self._cfg.args.eval_ood_out_circle_thr) + self._cfg.args.eval_ood_out_circle_thr
            out_samples = self._data['eval'].dataset.ambient_sampling(v=v_out)
            # Predict distance to the sample.
            pred_in_dist = (self._Gen.inverse(in_samples)[1][0]**2).sum(dim=(1, 2, 3)).sqrt()
            pred_out_dist = (self._Gen.inverse(out_samples)[1][0]**2).sum(dim=(1, 2, 3)).sqrt()
            """ Saving. """
            in_distances.append(pred_in_dist.cpu())
            out_distances.append(pred_out_dist.cpu())
        in_distances, out_distances = map(lambda _x: torch.cat(_x), [in_distances, out_distances])
        # Compute TRP95.
        metric = tpr95(in_distances, out_distances)
        return metric

    @torch.no_grad()
    def _eval_ood_eps(self):
        # Predict distance to manifold. (n_samples, )
        in_distances, out_distances = None, None
        for batch_data in self._data['eval']:
            batch_data = batch_data.to(self._cfg.args.device)
            # Get in-samples.
            in_samples = batch_data
            # Get out-samples.
            out_samples = batch_data + torch.rand_like(batch_data) * self._cfg.args.eval_ood_out_eps
            # Predict distance to the sample.
            pred_in_dist = torch.cat([(_v**2).sum(dim=(1, 2, 3))[:, None] for _v in self._Gen.inverse(in_samples)[1]], dim=1).sum(dim=1).sqrt().cpu()
            pred_out_dist = torch.cat([(_v**2).sum(dim=(1, 2, 3))[:, None] for _v in self._Gen.inverse(out_samples)[1]], dim=1).sum(dim=1).sqrt().cpu()
            """ Saving. """
            in_distances = pred_in_dist if in_distances is None else torch.cat([in_distances, pred_in_dist])
            out_distances = pred_out_dist if out_distances is None else torch.cat([out_distances, pred_out_dist])
            if len(in_distances) >= self._cfg.args.eval_ood_n_samples:
                in_distances = in_distances[:self._cfg.args.eval_ood_n_samples]
                out_distances = out_distances[:self._cfg.args.eval_ood_n_samples]
                break
        # Compute TPR95.
        metric = tpr95(in_distances, out_distances)
        return metric
