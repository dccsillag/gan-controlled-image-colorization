import os
import random
import numpy as np
import tensorflow as tf
from .options import ModelOptions
from .models import Cifar10Model, Places365Model
from .dataset import CIFAR10_DATASET, PLACES365_DATASET


def main(options):

    # reset tensorflow graph
    tf.reset_default_graph()


    # initialize random seed
    tf.set_random_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)


    # create a session environment
    with tf.Session() as sess:

        if options.dataset == CIFAR10_DATASET:
            model = Cifar10Model(sess, options)

        elif options.dataset == PLACES365_DATASET:
            print('model = Places365')
            model = Places365Model(sess, options)

        if not os.path.exists(options.checkpoints_path):
            os.makedirs(options.checkpoints_path)

        if options.log:
            open(model.train_log_file, 'w').close()
            open(model.test_log_file, 'w').close()

        # build the model and initialize
        model.build()
        sess.run(tf.global_variables_initializer())


        # load model only after global variables initialization
        model.load()


        if options.mode == 0:
            args = vars(options)
            print('\n------------ Options -------------')
            with open(os.path.join(options.checkpoints_path, 'options.dat'), 'w') as f:
                for k, v in sorted(args.items()):
                    print('%s: %s' % (str(k), str(v)))
                    f.write('%s: %s\n' % (str(k), str(v)))
            print('-------------- End ----------------\n')

            model.train()

        elif options.mode == 1:
            model.evaluate()
            while True:
                model.sample()

        elif options.mode == 2:
            model.turing_test()

        elif options.mode == 3:
            model.run_on_one_image()

        else:
            print("Invalid options.mode: %d" % options.mode)


if __name__ == "__main__":
    main(ModelOptions().parse())
