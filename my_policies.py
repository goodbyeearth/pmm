import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, my_conv, my_linear


def compute_old_conv(input=None, old_params=None, scop=None, n_fil=None, **kwargs):
    activ = tf.nn.relu
    old_conv = []
    for n in range(len(old_params)):
        param = old_params[n]
        # print("Use old model/" + scop + "/w & b")
        old_c = activ(
            my_conv(input[n], 'old_' + scop + str(n), n_filters=n_fil, filter_size=3, stride=1,
                    ww=param[scop + '/w'], bb=param[scop + '/b'], pad='SAME', **kwargs))
        if n == 0:
            sumc = old_c
        else:
            sumc = tf.add(sumc, old_c)
        old_conv.append(sumc)
    return old_conv, sumc


def compute_old_linear(input=None, old_params=None, scop=None):
    activ = tf.nn.relu
    old_linear = []
    for n in range(len(old_params)):
        param = old_params[n]
        # print("Use old model/" + scop + "/w & b")
        old_l = activ(my_linear(input[n], 'old_' + scop + str(n),
                                ww=param[scop + '/w'], bb=param[scop + '/b']))
        if n == 0:
            suml = old_l
        else:
            suml = tf.add(suml, old_l)
        old_linear.append(suml)
    return old_linear, suml


def custom_cnn_pgn(scaled_images, old_params=None, **kwargs):
    activ = tf.nn.relu

    layer_1 = activ(
        conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    old_conv = []
    for n in range(len(old_params)):
        param = old_params[n]
        # print(type(param))
        # print("Use old model/c1/w & b")
        old_c = activ(
            my_conv(scaled_images, 'old_c1' + str(n), n_filters=32, filter_size=3, stride=1,
                    ww=param['c1/w'], bb=param['c1/b'], pad='SAME', **kwargs))
        if n == 0:
            sumc = old_c
        else:
            sumc = tf.add(sumc, old_c)
        old_conv.append(sumc)

    layer_2 = activ(
        conv(tf.add(layer_1, sumc), 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME',
             **kwargs))
    old_conv, sumc = compute_old_conv(input=old_conv, old_params=old_params, n_fil=64, scop='c2')

    layer_3 = activ(
        conv(tf.add(layer_2, sumc), 'c3', n_filters=128, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME',
             **kwargs))
    old_conv, sumc = compute_old_conv(input=old_conv, old_params=old_params, n_fil=128, scop='c3')
    sumc = conv_to_fc(sumc)
    layer_3 = conv_to_fc(layer_3)
    for i in range(len(old_conv)):
        old_conv[i] = conv_to_fc(old_conv[i])

    layer_4 = activ(linear(tf.add(layer_3, sumc), 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    old_linear, suml = compute_old_linear(input=old_conv, scop='fc1', old_params=old_params)

    return layer_4, old_linear, suml


def custom_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(
        conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    layer_2 = activ(
        conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    layer_3 = activ(
        conv(layer_2, 'c3', n_filters=128, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, old_params=None, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        # print("Initial CustomPolicy...")

        with tf.variable_scope('model', reuse=reuse):
            activ = tf.nn.relu

            if old_params:
                print("Num of old networks", len(old_params))
                print()
                """CNN提取后的特征"""
                extracted_features, old_fc1, sum_fc1 = custom_cnn_pgn(self.processed_obs, old_params=old_params,
                                                                      **kwargs)
                extracted_features = tf.layers.flatten(extracted_features)
                for fc in range(len(old_fc1)):
                    old_fc1[fc] = tf.layers.flatten(old_fc1[fc])
                sum_fc1 = tf.layers.flatten(sum_fc1)

                pi_h = extracted_features
                layer_size = 128
                pi_h = activ(linear(tf.add(pi_h, sum_fc1), 'pi_fc0', n_hidden=layer_size))
                old_pi_fc0, sum_pi_fc0 = compute_old_linear(input=old_fc1, scop='pi_fc0', old_params=old_params)

                pi_h = activ(linear(tf.add(pi_h, sum_pi_fc0), 'pi_fc1', 64))
                old_pi_fc1, sum_pi_fc1 = compute_old_linear(input=old_pi_fc0, scop='pi_fc1', old_params=old_params)

                pi_latent = tf.add(pi_h, sum_pi_fc1)

                vf_h = extracted_features
                vf_h = activ(linear(tf.add(vf_h, sum_fc1), 'vf_fc0', n_hidden=layer_size))
                old_vf_fc0, sum_vf_fc0 = compute_old_linear(input=old_fc1, scop='vf_fc0', old_params=old_params)

                value_fn = linear(tf.add(vf_h, sum_vf_fc0), 'vf', n_hidden=1)
                old_vf, sum_vf = compute_old_linear(input=old_vf_fc0, scop='vf', old_params=old_params)

                vf_latent = tf.add(vf_h, sum_vf_fc0)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent_pgn(pi_latent, vf_latent, init_scale=0.01,
                                                                   old_pi_fc=old_pi_fc1, old_vf_fc=old_vf_fc0,
                                                                   old_params=old_params)

                self._value_fn = tf.add(value_fn, sum_vf)
                self._setup_init()
            else:
                print("No old networks")
                """CNN提取后的特征"""
                extracted_features = custom_cnn(self.processed_obs, **kwargs)
                extracted_features = tf.layers.flatten(extracted_features)

                pi_h = extracted_features
                # TODO: 调
                for i, layer_size in enumerate([128, 64]):
                    pi_h = activ(linear(pi_h, 'pi_fc' + str(i), n_hidden=layer_size))
                pi_latent = pi_h

                vf_h = extracted_features
                # TODO: 调
                for i, layer_size in enumerate([128]):
                    vf_h = activ(linear(vf_h, 'vf_fc' + str(i), n_hidden=layer_size))
                value_fn = linear(vf_h, 'vf', n_hidden=1)
                vf_latent = vf_h

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

                self._value_fn = value_fn
                self._setup_init()

    # TODO: 选取 deterministic 观察效果
    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
