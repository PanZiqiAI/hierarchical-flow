""" The main script for debugging. """

import config
from modellib.trainer import Trainer


if __name__ == '__main__':
    # 1. Generate config.
    cfg = config.ConfigTrain()
    # 2. Generate trainer & data.
    trainer = Trainer(cfg)
    # 3. Train
    trainer.eval_debug(n_samples=100)
