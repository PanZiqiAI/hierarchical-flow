
import os
from custom_pkg.io.config import CanonicalConfigPyTorch


########################################################################################################################
# Config for train.
########################################################################################################################

class ConfigTrain(CanonicalConfigPyTorch):
    """ The config for training the model. """
    def __init__(self, exp_dir=os.path.join(os.path.split(os.path.realpath(__file__))[0], "../STORAGE/experiments"), **kwargs):
        super(ConfigTrain, self).__init__(exp_dir, **kwargs)

    def _set_directory_args(self, **kwargs):
        super(ConfigTrain, self)._set_directory_args()
        self.args.eval_vis_dir = os.path.join(self.args.ana_train_dir, 'visuals')

    def _add_root_args(self):
        super(ConfigTrain, self)._add_root_args()
        self.parser.add_argument("--dataset",           type=str,   default='single-mnist')

    def _add_tree_args(self, args_dict):
        ################################################################################################################
        # Datasets.
        ################################################################################################################
        if args_dict['dataset'] == 'single-mnist':
            self.parser.set(['img_nc', 'img_size'], [1, 28])
            self.parser.add_argument("--dataset_category",      type=int,   default=8)
            self.parser.add_argument("--dataset_phase",         type=str,   default='train')
        if args_dict['dataset'] in ['celeba-hq', 'ffhq', 'afhq']:
            self.parser.set(['img_nc', 'img_size'], [3, 64])
            self.parser.add_argument("--dataset_maxsize",       type=int,   default=float("inf"))
        if args_dict['dataset'] == 'afhq':
            self.parser.add_argument("--dataset_category",      type=str,   default='cat')
            self.parser.add_argument("--dataset_phase",         type=str,   default='train')
        self.parser.add_argument("--dataset_drop_last",         type=bool,  default=True)
        self.parser.add_argument("--dataset_shuffle",           type=bool,  default=True)
        self.parser.add_argument("--dataset_num_threads",       type=int,   default=0)
        ################################################################################################################
        # Modules.
        ################################################################################################################
        self.parser.add_argument("--u_nc",                      type=int,   default=1)
        self.parser.add_argument("--middle_u_ncs",              type=int,   nargs='*',  default=[])
        self.parser.add_argument("--upsamples",                 type=int,   nargs='+',  default=[1])
        # Generator.
        self.parser.add_argument("--gen_hidden_ncs",            type=int,   nargs='+',  default=[1024])
        self.parser.add_argument("--gen_ns_couplings",          type=int,   nargs='+',  default=[3])
        # SV Predictor.
        self.parser.add_argument("--svp_hidden_ncs",            type=int,   nargs='+',  default=[1024])
        self.parser.add_argument("--svp_ns_layers",             type=int,   nargs='+',  default=[3])
        ################################################################################################################
        # Optimization.
        ################################################################################################################
        self.parser.add_argument("--sampling_u_mode",           type=str,   default='gauss',    choices=['gauss', 'uni'])
        self.parser.add_argument("--sampling_v_radius",         type=float, default=0.5)
        # Lambdas.
        # (1) Losses on reconstruction.
        self.parser.add_argument("--lambda_recon",              type=float, default=1.0)
        self.parser.add_argument("--recon_mode",                type=str,   default='l1')
        self.parser.add_argument("--lambda_v",                  type=float, default=1.0)
        # (2) Losses on Jacobian.
        self.parser.add_argument("--freq_step_jacob",           type=int,   default=1)
        self.parser.add_argument("--lambda_jacob",              type=float, default=1000.0)
        self.parser.add_argument("--sn_power",                  type=int,   default=3)
        ################################################################################################################
        # Evalulation.
        ################################################################################################################
        # Visualization.
        self.parser.add_argument("--freq_counts_step_eval_vis",     type=int,   default=100)
        self.parser.add_argument("--eval_recon_n_samples",          type=int,   default=10)
        # Jacobian.
        self.parser.add_argument("--freq_counts_step_eval_jacob",   type=int,   default=10)
        self.parser.add_argument("--eval_jacob_n_samples",          type=int,   default=32)
        self.parser.add_argument("--eval_jacob_batch_size",         type=int,   default=4)
        self.parser.add_argument("--eval_jacob_ag_bsize",           type=int,   default=24)

    def _add_additional_args(self):
        # Epochs & batch size
        self.parser.add_argument("--steps",                         type=int,   default=200000)
        self.parser.add_argument("--batch_size",                    type=int,   default=96)
        # Learning rate
        self.parser.add_argument("--learning_rate",                 type=float, default=0.0001)
        # Frequency
        self.parser.add_argument("--freq_counts_step_log",          type=int,   default=1000)
        self.parser.add_argument("--freq_counts_step_chkpt",        type=int,   default=50)
