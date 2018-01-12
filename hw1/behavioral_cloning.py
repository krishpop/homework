# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Krishnan Srinivasan <krishnan1994@gmail.com>
#
# Distributed under terms of the MIT license.
# ==============================================================================

"""
Implementation of behavior cloning.
"""

import gym
import os
import pickle
import tensorflow as tf
import tf_util
import utils

SAVE_PATH = './behavioral_cloning'

ACT_FNS = {'lrelu': tf_util.lrelu,
                  'relu': tf.nn.relu,
                  'softmax': tf.nn.softmax,
                  'softplus': tf.nn.softplus,
                  'sigmoid': tf.nn.sigmoid}

def load_behavior_cloning(envname, filename='best.config'):
    config_path = os.path.join(SAVE_PATH, envname, filename)
    config = None
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            config = pickle.loads(f.read())
        model = build_model_from_config(config)
    else:
        config = default_config(envname)
        model = build_model_from_config(config)


def default_config(envname):
    hidden_layers = [10, 10]
    env = gym.make(envname)
    obs_space_dim = env.observation_space.shape
    action_space_dim = env.action_space.shape[1]
    act_fn = 'lrelu'
    config = dict(hidden_layers=hidden_layers,envname=envname,
                  input_dim=obs_space_dim, output_dim=action_space_dim,
                  act_fn=act_fn, use_bias=True, lr=1e-3)
    return config


def build_model_from_config(config):
    return FeedForward(**config)


class FeedForward():
    def __init__(self, input_dim, output_dim, hidden_layers, act_fn, envname, use_bias, lr, save_path=SAVE_PATH):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_layers = hidden_layers
        self._act_fn = act_fn
        self._envname = envname
        self._use_bias = use_bias
        self._lr = float(lr)
        model_dict = dict(hidden_layers=','.join([str(x) for x in hidden_layers]), act_fn=act_fn,
                          envname=envname, use_bias=use_bias, lr=lr)
        self._model_str = utils.make_dict_str(model_dict, kv_sep='=', item_sep='-')
        self._save_path = os.path.join(save_path, envname, self._model_str)

    def restore_model(self, sess, ckpt_dir, ckpt_name='best.model'):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir, ckpt_name)
        if ckpt is None:
            return False
        else:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return True

    def build(self, sess):
        if self._built:
            print('Already built')
            return
        self.x_ = tf.placeholder(tf.float32, shape=[None, self._input_dim], name='input')
        self.y_ = tf.placeholder(tf.float32, shape=[None, self._output_dim], name='label')
        self.is_training_ = tf.placeholder_with_default(True, None, name='is_training')
        x_ = self.x_
        act_fn = ACT_FNS.get(self._act_fn, None)
        with tf.variable_scope('hidden_layers'):
            for i, h_dim in enumerate(self._hidden_layers):
                x_ = tf.layers.dense(x_, h_dim, act_fn, use_bias=self._use_bias)
                x_ = tf.layers.batch_normalization(x_, training=self.is_training_)
        out_dim = self._output_dim
        with tf.variable_scope('output'):
            self.prediction_ = tf.layers.dense(x_, out_dim, None, use_bias=self._use_bias)
        self.loss_ = tf.losses.softmax_cross_entropy(self.y_, self.prediction_)
        self.accuracy_ = tf.equal(tf.reduce_max(self.prediction_, axis=1), self.y_)
        self.optimizer = tf.train.AdamOptimizer(self._lr)
        self.global_step_ = tf.Variable(0, name='global_step' trainable=False)
        self.current_epoch_ = tf.Variable(0, name='current_epoch', trainable=False)
        self.opt_op_ = optimizer.minimize(self.loss_, self.global_step_)
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)
        if os.path.exists(self._save_path + '/best.model'):
            restored = self.restore_model(sess, self._save_path, 'best.model')
            tf.logging.info('Restored model: {}'.format(self._save_path + '/best.model'))
            tf.logging.info('Already trained for {} steps, {} epochs'.format(*sess.run([self.global_step_, self.current_epoch_])))
        return

    def save_model(self, sess, filenam, step=None):
        if filename != 'best.model':
            if len(self.saver.last_checkpoints) == 5:
                for ckpt in self.saver.last_checkpoints:
                    if ckpt != self._save_path + '/best.model':
                        break
                for ckpt_file in glob.glob(ckpt + '*'):
                    os.remove(ckpt_file)
                self.saver.last_checkpoints.remove(ckpt)
        self.saver.save(sess, self._save_path + '/{}'.format(filename), global_step=step)

    def train(self, sess, data, batch_size, num_steps, log_freq=100, ckpt_freq=1000, patience=None):
        self.best_test_loss = None
        self.epochs_trained = data.epochs_trained = self.current_epoch_.eval(sess)
        epochs_since_improved = 0
        steps_per_epoch = (data.num_samples // batch_size
                           + int(data.num_samples % batch_size != 0)
        saver = self.saver
        update_epochs_ph_ = tf.placeholder(tf.int32, name='update_epochs_ph')
        update_epochs = tf.assign(self.current_epoch_, update_epochs_ph_)
        epochs_trained = sess.run(self.current_epoch_)
        for step in range(num_steps):
            obs, true_actions = data.next_batch(batch_size)
            feed_dict = {self.x_: obs, self.y_: true_actions}
            predicted_actions, loss, _ = sess.run([self.prediction_, self.loss_, self.opt_op_], feed_dict=feed_dict)
            incorrect_predictions = np.sum(np.not_equal(true_actions, predicted_actions)) / 2
            log_str = ('epoch/step: {}/{}, '.self.(model.epochs_trained, step-1)
                + 'incorrect: {}'.format(incorrect_predictions) + 'loss: {}'.format(loss))
            tf.logging.log_first_n(tf.logging.INFO, log_str, log_freq - 1)
            tf.logging.log_every_n(tf.logging.INFO, log_str, log_freq)

            if epochs_trained != data.epochs_trained:
                epochs_trained = sess.run(update_epochs, feed_dict={update_epochs_ph_:data.epochs_trained})
                feed_dict = {self.x_: data.test_data, self.y_: data.test_labels,
                             self.is_training_: False}
                predicted_actions, loss = sess.run([self.prediction_, self.loss_], feed_dict=feed_dict)
                incorrect_predictions = np.sum(np.not_equal(true_actions, predicted_actions)) / 2
                log_str = ('TESTING -- epoch: {}, '.format(epochs_trained)
                           + 'incorrect: {} '.format(incorrect_predictions) + 'loss: {}'.format(loss))
                tf.logging.info(log_str)

                if self.best_test_loss is None or loss < self.best_test_loss:
                    epochs_since_improved = 0
                    self.best_test_loss = loss
                    saver.save(sess, self._save_path + '/best.model')
                    tf.logging.info('Best model saved after {} epochs'.format(epochs_trained))
                else:
                    epochs_since_improved += 1
                if patience and epochs_since_improved > patience:
                    tf.logging.info('Exiting training, not improved for {} epochs'.format(epochs_since_improved))

            if (step + 1) % ckpt_freq == 0:
                tf.logging.info('Saving model, after step {}'.format(step-1))
                self.save_model(sess, 'model', step=step) 
        return loss
