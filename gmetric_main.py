import sys

import numpy as np
import tensorflow as tf
from config import get_config
from trainer import Trainer


def main(_):
    #Set random seed
    tf.set_random_seed(123)

    #Model training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    config, unparsed = get_config()
    vars_config = vars(config)
    for keys, values in vars_config.items():
        print("%s : %s"%(keys, values))

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
