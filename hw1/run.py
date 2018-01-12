# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Krishnan Srinivasan <krishnan1994@gmail.com>
#
# Distributed under terms of the MIT license.
# ==============================================================================

"""
Train behavioral cloning model
"""

import behavioral_cloning as bc
import gym
import pickle
import numpy as np
import tensorflow as tf
import utils

from sklearn.cross_validation import train_test_split


tf.flags.DEFINE_string('envname', 'Ant-v1', 'name of environment')
tf.flags.DEFINE_string('model_config', None, 'name of model config file')
tf.flags.DEFINE_string('hidden_layers', '500,250', 'comma-separated list of layer shapes')
tf.flags.DEFINE_string('act_fn', 'lrelu', 'name of non-linearity to use in model')
tf.flags.DEFINE_boolean('use_bias', False, 'boolean for whether or not to use bias')
tf.flags.DEFINE_float('lr', 1e-3, 'optimizer learning rate')
tf.flags.DEFINE_integer('batch_size', 200, 'size of batch during training')
tf.flags.DEFINE_integer('num_epochs', 20, 'number of epochs to train')
tf.flags.DEFINE_integer('patience', None, 'number of epochs to train without improvement, early stopping')

FLAGS = tf.flags.FLAGS
RAND_SEED = 20
EXPERT_DATA = './expert_data'

def main(_):
    tf.set_random_seed(RAND_SEED)
    FLAGS.hidden_layers = [x.strip() for x in FLAGS.hidden_layers.split(',')]
    if FLAGS.model_config:
        model = bc.load_behavioral_cloning(FLAGS.envname, FLAGS.model_config)
    else:
        env = gym.make(envname)
        obs_space_dim = env.observation_space.shape
        action_space_dim = env.action_space.shape[1]
        config = dict(hidden_layers=FLAGS.hidden_layers,envname=FLAGS.envname,
                      input_dim=obs_space_dim, output_dim=action_space_dim,
                      act_fn=FLAGS.act_fn, zero_bias=FLAGS.zero_bias, lr=FLAGS.lr)
        model = bc.build_model_from_config(config)
    save_file = os.path.join(EXPERT_DATA, FLAGS.envname + '.npz')
    data = DataSet.load(save_file)
    num_steps = data.num_samples // FLAGS.batch_size * FLAGS.num_epochs

    with tf.get_default_session() as sess:
        model.build(sess)
        model.train(sess, data, FLAGS.batch_size, num_steps, FLAGS.patience)

    return


if __name__ == '__main__':
    tf.app.run()
