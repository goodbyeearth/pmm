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
                    ww=param[scop + '/w'], bb=param[scop + '/b'], **kwargs))
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

    layer_1 = activ(conv(scaled_images, 'c1', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    # old_conv,sumc = compute_old_conv(input=scaled_images,old_params=old_params,scop='c1')
    old_conv = []
    for n in range(len(old_params)):
        param = old_params[n]
        # print(type(param))
        # print("Use old model/c1/w & b")
        old_c = activ(
            my_conv(scaled_images, 'old_c1' + str(n), n_filters=16, filter_size=3, stride=1,
                    ww=param['c1/w'], bb=param['c1/b'], **kwargs))
        if n == 0:
            sumc = old_c
        else:
            sumc = tf.add(sumc, old_c)
        old_conv.append(sumc)

    layer_2 = activ(
        conv(tf.add(layer_1, sumc), 'c2', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    old_conv, sumc = compute_old_conv(input=old_conv, old_params=old_params, n_fil=32, scop='c2')
    # old_conv2 = []
    # for n in range(len(old_params)):
    #     param = old_params[n]
    #     print("Use old model/c2/w & b")
    #     old_c = activ(my_conv(old_conv1[n], 'old_c2', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2),
    #                           ww=param['c2/w'], bb=old_params['c2/b'], **kwargs))
    #     old_conv2.append(old_c)
    #     print()
    # flag = True
    # for c in old_conv2:
    #     if flag:
    #         sum2 = c
    #         flag = False
    #     else:
    #         sum2 = tf.add(sum2, c)

    layer_3 = activ(
        conv(tf.add(layer_2, sumc), 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    old_conv, sumc = compute_old_conv(input=old_conv, old_params=old_params, n_fil=64, scop='c3')
    # old_conv3 = []
    #     # for n in range(len(old_params)):
    #     #     param = old_params[n]
    #     #     print("Use old model/c3/w & b")
    #     #     old_c = activ(my_conv(old_conv2[n], 'old_c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2),
    #     #                           ww=param['c3/w'], bb=old_params['c3/b'], **kwargs))
    #     #     old_conv3.append(old_c)
    #     #     print()
    #     # flag = True
    #     # for c in old_conv3:
    #     #     if flag:
    #     #         sum3 = c
    #     #         flag = False
    #     #     else:
    #     #         sum3 = tf.add(sum3, c)
    sumc = conv_to_fc(sumc)
    layer_3 = conv_to_fc(layer_3)
    for i in range(len(old_conv)):
        old_conv[i] = conv_to_fc(old_conv[i])

    layer_4 = activ(linear(tf.add(layer_3, sumc), 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    old_linear, suml = compute_old_linear(input=old_conv, scop='fc1', old_params=old_params)
    # old_linear = []
    # for n in range(len(old_params)):
    #     param = old_params[n]
    #     print("Use old model/fc1/w & b")
    #     old_l = activ(my_linear(old_conv3[n], 'old_fc1', n_hidden=256, init_scale=np.sqrt(2),
    #                             ww=param['fc1/w'], bb=old_params['fc1/b']))
    #     old_linear.append(old_l)
    #     print()
    # flag = True
    # for l in old_linear:
    #     if flag:
    #         sum4 = l
    #         flag = False
    #     else:
    #         sum4 = tf.add(sum4, l)

    return layer_4, old_linear, suml


def custom_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
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
                extracted_features, old_fc1, sum_fc1 = custom_cnn_pgn(self.processed_obs, old_params=old_params, **kwargs)
                extracted_features = tf.layers.flatten(extracted_features)
                for fc in range(len(old_fc1)):
                    old_fc1[fc] = tf.layers.flatten(old_fc1[fc])
                sum_fc1 = tf.layers.flatten(sum_fc1)

                pi_h = extracted_features
                layer_size = 64
                pi_h = activ(linear(tf.add(pi_h, sum_fc1), 'pi_fc0', n_hidden=layer_size))
                old_pi_fc0, sum_pi_fc0 = compute_old_linear(input=old_fc1, scop='pi_fc0', old_params=old_params)
                # old_pi_fc0 = []
                # for n in range(len(old_params)):
                #     param = old_params[n]
                #     print('Use old model/pi_fc0/w & b')
                #     old_l = activ(
                #         my_linear(old_fc1[n], 'old_pi_fc0' + str(n), n_hidden=layer_size, ww=param['pi_fc0/w'],
                #                   bb=param['pi_fc0/b']))
                #     old_pi_fc0.append(old_l)
                #     print()
                # flag = True
                # for l in old_pi_fc0:
                #     if flag:
                #         sum_pi_fc0 = l
                #         flag = False
                #     else:
                #         sum_pi_fc0 = tf.add(sum_pi_fc0, l)
                pi_h = activ(linear(tf.add(pi_h, sum_pi_fc0), 'pi_fc1', layer_size))
                old_pi_fc1, sum_pi_fc1 = compute_old_linear(input=old_pi_fc0, scop='pi_fc1', old_params=old_params)
                # old_pi_fc1 = []
                # for n in range(len(old_params)):
                #     param = old_params[n]
                #     print('Use old model/pi_fc1/w & b')
                #     old_l = activ(
                #         my_linear(old_pi_fc0[n], 'old_pi_fc1' + str(n), n_hidden=layer_size, ww=param['pi_fc1/w'],
                #                   bb=param['pi_fc1/b']))
                #     old_pi_fc1.append(old_l)
                #     print()
                # flag = True
                # for l in old_pi_fc1:
                #     if flag:
                #         sum_pi_fc1 = l
                #         flag = False
                #     else:
                #         sum_pi_fc1 = tf.add(sum_pi_fc1, l)
                pi_latent = tf.add(pi_h, sum_pi_fc1)
                
                vf_h = extracted_features
                vf_h = activ(linear(tf.add(vf_h, sum_fc1), 'vf_fc0', n_hidden=layer_size))
                old_vf_fc0, sum_vf_fc0 = compute_old_linear(input=old_fc1, scop='vf_fc0', old_params=old_params)
                # old_vf_fc = []
                # for i, layer_size in enumerate([64]):
                #     vf_h = activ(linear(tf.add(vf_h, sum_fc1), 'vf_fc' + str(i), n_hidden=layer_size))
                #     for n in range(len(old_params)):
                #         param = old_params[n]
                #         print('Use old model/vf_fc0/w & b')
                #         old_l = activ(
                #             my_linear(old_fc1[n], 'old_vf_fc0' + str(n), n_hidden=layer_size, ww=param['vf_fc0/w'],
                #                       bb=param['vf_fc0/b']))
                #         old_vf_fc.append(old_l)
                #         print()
                #     flag = True
                #     for l in old_vf_fc:
                #         if flag:
                #             sum_vf_fc = l
                #             flag = False
                #         else:
                #             sum_vf_fc = tf.add(sum_vf_fc, l)
                value_fn = linear(tf.add(vf_h, sum_vf_fc0), 'vf', n_hidden=1)
                old_vf, sum_vf = compute_old_linear(input=old_vf_fc0, scop='vf', old_params=old_params)
                # old_vf = []
                # for n in range(len(old_params)):
                #     param = old_params[n]
                #     print('Use old model/vf/w & b')
                #     old_l = activ(my_linear(old_vf_fc[n], 'old_vf' + str(n), n_hidden=layer_size, ww=param['vf/w'],
                #                             bb=param['vf/b']))
                #     old_vf.append(old_l)
                #     print()
                # flag = True
                # for l in old_vf:
                #     if flag:
                #         sum_vf = l
                #         flag = False
                #     else:
                #         sum_vf = tf.add(sum_vf, l)
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
                for i, layer_size in enumerate([64, 64]):
                    pi_h = activ(linear(pi_h, 'pi_fc' + str(i), n_hidden=layer_size))
                pi_latent = pi_h

                vf_h = extracted_features
                # TODO: 调
                for i, layer_size in enumerate([64]):
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
