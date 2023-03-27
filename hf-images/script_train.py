""" The main script for training. """

import config
from dataloader import generate_data
from modellib.trainer import Trainer


if __name__ == '__main__':
    # 1. Generate config.
    cfg = config.ConfigTrain()
    # 2. Generate trainer & data.
    trainer = Trainer(cfg)
    train_data = generate_data(cfg)
    eval_data = generate_data(cfg)
    # 3. Train
    trainer.train_model(train_data=train_data, eval_data=eval_data)
