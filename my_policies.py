import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc


def custom_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    # TODO: 调
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=5, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    # layer_3 = conv_to_fc(layer_2)
    return activ(linear(layer_3, 'fc1', n_hidden=1024, init_scale=np.sqrt(2)))


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        # print("Initial CustomPolicy...")

        with tf.variable_scope('model', reuse=reuse):
            activ = tf.nn.relu

            """CNN提取后的特征"""
            extracted_features = custom_cnn(self.processed_obs, **kwargs)
            extracted_features = tf.layers.flatten(extracted_features)

            pi_h = extracted_features
            # TODO: 调
            for i, layer_size in enumerate([512, 512, 256, 128, 32]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc'+str(i)))
            pi_latent = pi_h

            vf_h = extracted_features
            # TODO: 调
            for i, layer_size in enumerate([512, 256, 64, 32]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc'+str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
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
