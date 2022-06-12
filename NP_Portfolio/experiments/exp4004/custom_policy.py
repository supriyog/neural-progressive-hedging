import errno
import os

from ray.experimental.tf_utils import TensorFlowVariables
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.custom_policy import CustomTFPolicy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override
from ray.rllib.utils.tf_ops import scope_vars
tf = try_import_tf()
import tensorflow as tf_old

import numpy as np

POLICY_SCOPE = 'policy'

class CustomPolicy(CustomTFPolicy):

    def __init__(self, observation_space, action_space, config):
        config = dict(COMMON_CONFIG, **config)
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.observation = tf.placeholder(
            tf.float32, shape=(None,)+observation_space.shape, name="obs")
        self.next_observation = tf.placeholder(
            tf.float32, shape=(None,)+observation_space.shape, name="next_obs")
        with tf.variable_scope(POLICY_SCOPE) as scope:
            self.policy  = self._build_policy_network(
                self.observation, observation_space, action_space)
            self.policy_vars = scope_vars(scope.name)
        self.loss = self._build_loss(self.policy, self.observation, self.next_observation)
        self.global_step = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
        self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step, var_list=self.policy_vars)
        self.grads_and_vars = [(g,v) for (g,v) in self.optimizer.compute_gradients(self.loss, var_list=self.policy_vars) if g is not None]
        self.sess = tf.get_default_session() or tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.variables = TensorFlowVariables(self.policy, self.sess)

    @override(TFPolicy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        act_batch = self.sess.run(
            self.policy,
            feed_dict={
                self.observation: obs_batch,
                self._get_is_training_placeholder(): False,
            })
        return act_batch, [], {}

    @override(TFPolicy)
    def learn_on_batch(self, samples):
        obs_batch, next_obs_batch = [
            list(x) for x in samples.columns(
                [SampleBatch.CUR_OBS, SampleBatch.NEXT_OBS])
        ]
        if self.config['dump_gradients']:
            grads_and_vars = self.sess.run(
                self.grads_and_vars,
                feed_dict={
                    self.observation: obs_batch,
                    self.next_observation: next_obs_batch,
                })
        else:
            grads_and_vars = None
        results = self.sess.run(
            [self.train_step, self.policy], 
            feed_dict={
                self.observation: obs_batch,
                self.next_observation: next_obs_batch,
            })
        v = []
        for next_obs, act in zip(next_obs_batch, results[-1]):
            next_obs[:,-1,:] = np.expand_dims(act, axis=1)
            v.append(next_obs)
        return {
            LEARNER_STATS_KEY: {},
            'next_obs_batch': v,
            'grads_and_vars': grads_and_vars,
        }

    @override(TFPolicy)
    def get_weights(self):
        return self.variables.get_weights()

    @override(TFPolicy)
    def set_weights(self, weights):
        self.variables.set_weights(weights)

    @override(TFPolicy)
    def export_checkpoint(self, export_dir, filename_prefix="model"):
        try:
            os.makedirs(export_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        save_path = os.path.join(export_dir, filename_prefix)
        with self.sess.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, save_path, write_meta_graph=False)

    def _get_is_training_placeholder(self):
        if not hasattr(self, "_is_training"):
            self._is_training = tf.placeholder_with_default(False, ())
        return self._is_training

    def evaluate(self, obs_batch, **kwargs):
        return self.sess.run(
            self.policy,
            feed_dict={
                self.observation: obs_batch,
                self._get_is_training_placeholder(): False,
            })

    def _build_loss(self, policy, obs, next_obs):
        future_price = next_obs[:,:,-2,3] / next_obs[:,:,-3,3]
        future_omega = (future_price * policy) / tf.reduce_sum(future_price * policy, axis=1)[:,None]
        w_t = future_omega[:-1]
        w_t1 = policy[1:]
        mu = 1 - tf.reduce_sum(tf.abs(w_t1[:,1:] - w_t[:,1:]), axis=1) * self.config['custom_policy_config']['trading_cost']
        mu = tf.concat([tf.ones(1), mu], axis=0)
        pv_vector = tf.reduce_sum(policy * future_price, reduction_indices=[1]) * mu
        reg_losses = tf.losses.get_regularization_loss()
        return -tf.reduce_mean(tf.log(pv_vector)) + reg_losses

    def _build_policy_network(self, obs, obs_space, action_space):

        # ========== CONFIGURATION ========== #

        obs_norm_epsilon = 1e-12
        obs_norm_momentum = 0.99
        act_norm_epsilon = 1e-12
        act_norm_momentum = 0.99
        #predictor_type = 'cnn'
        #predictor_hiddens = [3,10]
        #predictor_filters = [2,None]
        #predictor_type = 'lstm'
        #predictor_hiddens = [20,8]
        #predictor_filters = None
        predictor_can_perturbs = False
        predictor_use_batch_norms = False
        predictor_use_layer_norms = False
        predictor_batch_norm_epsilon = 1e-12
        predictor_batch_norm_momentum = 0.99
        #predictor_regularizer_weights = [0,5e-8] #[1e-2,1e-2] #[0,5e-9]
        predictor_initializer_schemes = ['uniform_scaling','uniform_scaling']
        predictor_dropout_rate = 0
        regularizer_weight = 0 #1e-2 #5e-8
        initializer_scheme = 'uniform_scaling' # HE or lillicrap
        #final_regularizer_weight = 5e-7 #1e-2 #5e-8
        final_initializer_scheme = 'uniform_scaling' # HE or lillicrap
        eiie_hiddens = []
        hiddens = [] #[256,128,64,32]
        #cash_bias_trainable = True
        use_batch_norm = False
        use_layer_norm = False
        batch_norm_epsilon = 1e-12
        batch_norm_momentum = 0.99

        activation = self.config['custom_policy_config']['activation']
        separate_cash = self.config['custom_policy_config']['separate_cash']
        cash_bias_trainable = self.config['custom_policy_config']['cash_bias_trainable']
        predictor_type = self.config['custom_policy_config']['predictor_type']
        predictor_hiddens = self.config['custom_policy_config']['predictor_hiddens']
        predictor_filters = self.config['custom_policy_config']['predictor_filters']
        predictor_regularizer_weights = self.config['custom_policy_config']['predictor_regularizer_weights']
        final_regularizer_weight = self.config['custom_policy_config']['final_regularizer_weight']
        rebalance_cash = self.config['custom_policy_config']['rebalance_cash']

        predictor_kernel_regularizers = [
            None if l == 0 else tf_old.contrib.layers.l2_regularizer(scale=l)
            for l in predictor_regularizer_weights
        ]
        #predictor_bias_regularizers = [
        #    None if l == 0 else tf_old.contrib.layers.l2_regularizer(scale=l)
        #    for l in predictor_regularizer_weights
        #]
        predictor_bias_regularizers = [
            None for l in predictor_regularizer_weights
        ]
        predictor_kernel_initializers = [
            (tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal', seed=None, dtype=tf.float32) if s == 'HE' else
             tf.variance_scaling_initializer(scale=1/3, mode='fan_in', distribution='uniform', seed=None, dtype=tf.float32) if s == 'lillicrap' else
             tf.uniform_unit_scaling_initializer(seed=None, dtype=tf.float32) if s == 'uniform_scaling' else
             None)
            for s in predictor_initializer_schemes
        ]
        predictor_bias_initializers = [
            tf.zeros_initializer(dtype=tf.float32) 
            for s in predictor_initializer_schemes
        ]

        kernel_regularizer = None if regularizer_weight == 0 else tf_old.contrib.layers.l2_regularizer(scale=regularizer_weight)
        #bias_regularizer = None if regularizer_weight == 0 else tf_old.contrib.layers.l2_regularizer(scale=regularizer_weight)
        bias_regularizer = None 
        kernel_initializer = (
            tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal', seed=None, dtype=tf.float32) if initializer_scheme == 'HE' else
            tf.random_uniform_initializer(minval=-0.0003, maxval=+0.0003, seed=None, dtype=tf.float32) if initializer_scheme == 'lillicrap' else 
            tf.uniform_unit_scaling_initializer(seed=None, dtype=tf.float32) if initializer_scheme == 'uniform_scaling' else
            None
        )
        bias_initializer = tf.zeros_initializer(dtype=tf.float32) 

        final_kernel_regularizer = None if final_regularizer_weight == 0 else tf_old.contrib.layers.l2_regularizer(scale=final_regularizer_weight)
        #final_bias_regularizer = None if final_regularizer_weight == 0 else tf_old.contrib.layers.l2_regularizer(scale=final_regularizer_weight)
        final_bias_regularizer = None 
        final_kernel_initializer = (
            tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal', seed=None, dtype=tf.float32) if final_initializer_scheme == 'HE' else
            tf.random_uniform_initializer(minval=-0.0003, maxval=+0.0003, seed=None, dtype=tf.float32) if final_initializer_scheme == 'lillicrap' else 
            tf.uniform_unit_scaling_initializer(seed=None, dtype=tf.float32) if final_initializer_scheme == 'uniform_scaling' else
            None
        )
        final_bias_initializer = tf.zeros_initializer(dtype=tf.float32) 

        # ========== INPUTS ========== #

        if separate_cash:
            prev_actions = obs[:,1:,-1:,:1]
        else:
            prev_actions = obs[:,:,-1:,:1]
        obs = preprocessor(
            obs, separate_cash=separate_cash, normalize_by=None, use_log=False, scale=100, use_batch_norm=False, 
            training=self._get_is_training_placeholder(), 
            obs_norm_epsilon=obs_norm_epsilon, obs_norm_momentum=obs_norm_momentum)

        #net2 = tf.transpose(prev_actions, [0,2,3,1])
        #net2 = tf_old.contrib.layers.batch_norm(
        #    net2, is_training=self._get_is_training_placeholder(), scale=True, reuse=tf.get_variable_scope().reuse,
        #    epsilon=act_norm_epsilon, decay=act_norm_momentum, scope='act_batch_norm'
        #)
        #net2 = tf.transpose(net2, [0,3,1,2])
        net2 = prev_actions

        # ========== NETWORK ========== #

        num_stocks = obs.get_shape()[1]

        net = stock_predictor(
            obs, predictor_type, predictor_hiddens, activation=activation, filters=predictor_filters, 
            can_perturbs=predictor_can_perturbs,
            use_batch_norms=predictor_use_batch_norms, 
            use_layer_norms=predictor_use_layer_norms,
            batch_norm_epsilon=predictor_batch_norm_epsilon, batch_norm_momentum=predictor_batch_norm_momentum,
            kernel_initializers=predictor_kernel_initializers, bias_initializers=predictor_bias_initializers,
            kernel_regularizers=predictor_kernel_regularizers, bias_regularizers=predictor_bias_regularizers,
            training=self._get_is_training_placeholder(), dropout_rate=predictor_dropout_rate,
        )

        net = tf.concat([net,net2], axis=3)

        with tf.variable_scope('perturbable') as scope:

            if len(eiie_hiddens) > 0:

                num_channels = net.get_shape()[-1]

                for k,units in enumerate(eiie_hiddens):

                    conv_kernel = tf.get_variable(
                        'fa_eiie_dense_{}_conv_kernel'.format(k), [1,1,num_channels,units], trainable=True,
                        regularizer=final_kernel_regularizer, initializer=final_kernel_initializer, dtype=tf.float32,
                    )
                    conv_bias = tf.get_variable(
                        'fa_eiie_dense_{}_conv_bias'.format(k), [units], trainable=True, 
                        regularizer=final_bias_regularizer, initializer=final_bias_initializer, dtype=tf.float32, 
                    )
                    net = tf.nn.conv2d(net, conv_kernel, [1,1,1,1], padding="VALID") 
                    net = tf.nn.bias_add(net, conv_bias)

                    if use_batch_norm:
                        net = tf_old.contrib.layers.batch_norm(
                            net, is_training=training, scale=False, reuse=tf.get_variable_scope().reuse,
                            epsilon=batch_norm_epsilon, decay=batch_norm_momentum, scope='fa_eiie_dense_{}_batch_norm'.format(k)
                        )
                    if use_layer_norm:
                        net = tf_old.contrib.layers.layer_norm(
                            net, scale=False, reuse=tf.get_variable_scope().reuse, scope='fa_eiie_dense_{}_layer_norm'.format(k)
                        )

                    net = tf.nn.relu(net)
                    net = tf.nn.dropout(net, rate=0) #self._get_dropout_rate_placeholder())

                    num_channels = units

            if len(hiddens) == 0:

                num_channels = net.get_shape()[-1]

                conv_kernel = tf.get_variable(
                    'fa_eiie_output_conv_kernel', [1,1,num_channels,1], trainable=True,
                    regularizer=final_kernel_regularizer, initializer=final_kernel_initializer, dtype=tf.float32,
                )
                conv_bias = tf.get_variable(
                    'fa_eiie_output_conv_bias', [1], trainable=True, 
                    regularizer=final_bias_regularizer, initializer=final_bias_initializer, dtype=tf.float32, 
                )
                net = tf.nn.conv2d(net, conv_kernel, [1,1,1,1], padding="VALID") 
                net = tf.nn.bias_add(net, conv_bias)

                net = tf_old.contrib.layers.flatten(net)

                if separate_cash:
                    if cash_bias_trainable:
                        cash_bias = tf.tile(
                            tf.get_variable(
                                'fa_output_cash_bias', [1,1], trainable=cash_bias_trainable,
                                dtype=tf.float32, initializer=tf.zeros_initializer),
                            [tf.shape(obs)[0],1]
                        )
                    else:
                        cash_bias = tf.zeros([tf.shape(obs)[0],1])
                    net = tf.concat([cash_bias, net], 1)

            else:

                net = tf_old.contrib.layers.flatten(net)

                width = net.get_shape()[1]

                for k,units in enumerate(hiddens):

                    W = tf.get_variable(
                        'fa_{}_W'.format(k), [width,units], trainable=True,
                        regularizer=kernel_regularizer, initializer=kernel_initializer, dtype=tf.float32,
                    )
                    b = tf.get_variable(
                        'fa_{}_b'.format(k), [units], trainable=True, 
                        regularizer=bias_regularizer, initializer=bias_initializer, dtype=tf.float32, 
                    )
                    net = tf.nn.bias_add(tf.matmul(net, W), b)
                    if use_batch_norm:
                        net = tf_old.contrib.layers.batch_norm(
                            net, is_training=self._get_is_training_placeholder(), scale=False, reuse=tf.get_variable_scope().reuse,
                            epsilon=batch_norm_epsilon, decay=batch_norm_momentum, scope='fa_{}_batch_norm'.format(k)
                        )
                    if use_layer_norm:
                        net = tf_old.contrib.layers.layer_norm(
                            net, scale=False, reuse=tf.get_variable_scope().reuse, scope='fa_{}_layer_norm'.format(k)
                        )
                    net = tf.nn.relu(net)
                    net = tf.nn.dropout(net, rate=0) #self._get_dropout_rate_placeholder())
                    width = units

                W = tf.get_variable(
                    'fa_output_W', [width,action_space.shape[0]], trainable=True,
                    regularizer=final_kernel_regularizer, initializer=final_kernel_initializer, dtype=tf.float32,
                )
                b = tf.get_variable(
                    'fa_output_b', [action_space.shape[0]], trainable=True, 
                    regularizer=final_bias_regularizer, initializer=final_bias_initializer, dtype=tf.float32, 
                )
                net = tf.nn.bias_add(tf.matmul(net, W), b)

            if rebalance_cash:
                out = tf.nn.softmax(net)
            else:
                out = tf.concat([tf.zeros([tf.shape(obs)[0],1]),tf.nn.softmax(net[:,1:])], axis=1)

            #eps_crp_weight = self.config['custom_ddpg_config']['actor']['eps_crp_weight']
            #if eps_crp_weight == 'trainable':
            #    eps_crp_baseline = self.config['custom_ddpg_config']['actor']['eps_crp_baseline']
            #    if eps_crp_baseline == 'eq':
            #        base_out = tf.tile(
            #                tf.constant([[1/action_space.shape[0]]], dtype=tf.float32),
            #                tf.constant([1,action_space.shape[0]], dtype=tf.int32)
            #            )
            #    elif isinstance(eps_crp_baseline, int):
            #        crp_weights = np.load('data/crp_weights_v1_00_99.npy')
            #        base_out = tf.constant(crp_weights[eps_crp_baseline], dtype=tf.float32)
            #    else:
            #        raise
            #    eps_crp_weight = tf.nn.sigmoid(tf.get_variable(
            #        'eps_crp_weight', [1], initializer=tf.zeros_initializer(dtype=tf.float32),
            #        trainable=True, dtype=tf.float32))
            #    out = (1 - eps_crp_weight) * out + eps_crp_weight * base_out
            #    pre_softmax = tf.math.log(tf.maximum(0.0001, out))
            #elif eps_crp_weight > 0:
            #    eps_crp_baseline = self.config['custom_ddpg_config']['actor']['eps_crp_baseline']
            #    if eps_crp_baseline == 'eq':
            #        base_out = tf.tile(
            #                tf.constant([[1/action_space.shape[0]]], dtype=tf.float32),
            #                tf.constant([1,action_space.shape[0]], dtype=tf.int32)
            #            )
            #            #np.array([[+1/num_stocks,+1/7,+1/7,+1/7,+1/7,+1/7,+1/7]]), dtype=tf.float32)
            #    elif isinstance(eps_crp_baseline, int):
            #        crp_weights = np.load('data/crp_weights_v1_00_99.npy')
            #        base_out = tf.constant(crp_weights[eps_crp_baseline], dtype=tf.float32)
            #    else:
            #        raise
            #    out = (1 - eps_crp_weight) * out + eps_crp_weight * base_out
            #    pre_softmax = tf.math.log(tf.maximum(0.0001, out))

        return out


def stock_predictor(
        obs, predictor_type, hiddens, activation='relu', filters=None,
        can_perturbs=False, use_batch_norms=False, use_layer_norms=False,
        batch_norm_epsilon=1e-5, batch_norm_momentum=0.99,
        kernel_initializers=None, bias_initializers=None,
        kernel_regularizers=None, bias_regularizers=None, 
        training=False, dropout_rate=0.0,
    ):

    can_perturbs = (
        can_perturbs if isinstance(can_perturbs, list) else
        [can_perturbs for _ in range(len(hiddens))]
    )
    use_batch_norms = (
        use_batch_norms if isinstance(use_batch_norms, list) else
        [use_batch_norms for _ in range(len(hiddens))]
    )
    use_layer_norms = (
        use_layer_norms if isinstance(use_layer_norms, list) else
        [use_layer_norms for _ in range(len(hiddens))]
    )

    kernel_initializers = kernel_initializers or [
        tf.variance_scaling_initializer(
            scale=1/3, mode='fan_in', distribution='uniform', 
            seed=None, dtype=tf.float32,
        ) for _ in range(len(hiddens))
    ]
    bias_initializers = bias_initializers or [
        tf.zeros_initializer(dtype=tf.float32) for _ in range(len(hiddens))
    ]

    kernel_regularizers = kernel_regularizers or [
        None for _ in range(len(hiddens))
    ]
    bias_regularizers = bias_regularizers or [
        None for _ in range(len(hiddens))
    ]

    if predictor_type == 'cnn':
        net = obs
        num_stocks = obs.get_shape()[1]
        window_length = obs.get_shape()[2]
        num_features = obs.get_shape()[3]
        num_channels = None
        length = window_length
        for k, (units, width, 
                can_perturb, use_batch_norm, use_layer_norm,
                kernel_initializer, bias_initializer, 
                kernel_regularizer, bias_regularizer) in enumerate(zip(
                    hiddens, filters, 
                    can_perturbs, use_batch_norms, use_layer_norms,
                    kernel_initializers, bias_initializers, 
                    kernel_regularizers, bias_regularizers,
            )):
            if width is None:
                width = length
            if num_channels is None:
                num_channels = num_features
            if can_perturb:
                scope_name = 'perturbable'
            else:
                scope_name = 'fixed'
            with tf.variable_scope(scope_name) as scope:
                conv_kernel = tf.get_variable(
                    'fe_{}_conv_kernel'.format(k), [1,int(width),num_channels,units], trainable=True,
                    regularizer=kernel_regularizer, initializer=kernel_initializer, dtype=tf.float32,
                )
                conv_bias = tf.get_variable(
                    'fe_{}_conv_bias'.format(k), [units], trainable=True, 
                    regularizer=bias_regularizer, initializer=bias_initializer, dtype=tf.float32, 
                )
                net = tf.nn.conv2d(net, conv_kernel, [1,1,1,1], padding="VALID") 
                net = tf.nn.bias_add(net, conv_bias)
                if use_batch_norm:
                    net = tf_old.contrib.layers.batch_norm(
                        net, is_training=training, scale=False, reuse=tf.get_variable_scope().reuse,
                        epsilon=batch_norm_epsilon, decay=batch_norm_momentum, scope='fe_{}_batch_norm'.format(k)
                    )
                if use_layer_norm:
                    net = tf_old.contrib.layers.layer_norm(
                        net, scale=True, reuse=tf.get_variable_scope().reuse, scope='fe_{}_layer_norm'.format(k)
                    )
            if activation == 'relu':
                net = tf.nn.relu(net)
            elif activation == 'sigmoid':
                net = tf.nn.sigmoid(net)
            else:
                raise
            net = tf.nn.dropout(net, rate=dropout_rate)
            num_channels = units
            length = length - width + 1
    elif predictor_type == 'lstm':
        num_stocks = obs.get_shape()[1]
        window_length = obs.get_shape()[2]
        num_features = obs.get_shape()[3]
        net = tf.reshape(obs, [-1, window_length, num_features])
        for k, (units, use_layer_norm) in enumerate(zip(hiddens, use_layer_norms)):
            cell = tf_old.contrib.rnn.LayerNormBasicLSTMCell(
                units, reuse=tf.get_variable_scope().reuse, dropout_keep_prob=1-dropout_rate, layer_norm=use_layer_norm)
            #cell = tf.nn.rnn_cell.LSTMCell(
            #    units, reuse=tf.get_variable_scope().reuse, state_is_tuple=False, name='fe_{}_lstm'.format(k))
            net, state = tf.nn.dynamic_rnn(cell, net, dtype=tf.float32, scope='fe_{}_lstm'.format(k))
        net = net[:,-1,:]
        net = tf.reshape(net, [-1, num_stocks, hiddens[-1]])
        net = tf.expand_dims(net, axis=2)
    elif predictor_type == 'rnn':
        num_stocks = obs.get_shape()[1]
        window_length = obs.get_shape()[2]
        num_features = obs.get_shape()[3]
        net = tf.reshape(obs, [-1, window_length, num_features])
        for k, units in enumerate(hiddens):
            cell = tf_old.nn.rnn_cell.BasicRNNCell(units, reuse=tf.get_variable_scope().reuse)
            #cell = tf_old.contrib.cudnn_rnn.CudnnRNNTanh(units, reuse=tf.get_variable_scope().reuse)
            net, state = tf.nn.dynamic_rnn(cell, net, dtype=tf.float32, scope='fe_{}_rnn'.format(k))
        net = net[:,-1,:]
        net = tf.reshape(net, [-1, num_stocks, hiddens[-1]])
        net = tf.expand_dims(net, axis=2)
    else:
        raise NotImplementedError
    return net


def preprocessor(obs, separate_cash=True, normalize_by=None, use_log=False, scale=1, use_batch_norm=False, training=False, obs_norm_epsilon=1e-5, obs_norm_momentum=0.99):

    #obs = obs[:,1:,:-1,3:4]
    if separate_cash:
        obs = obs[:,1:,:-1,1:4]
    else:
        obs = obs[:,:,:-1,1:4]
    obs = obs / obs[:,:,-1,-1,None,None]

    if normalize_by is not None:
        obs = normalize_by * (obs - 1)

    if use_log:
        obs = scale * tf.math.log(obs)

    if use_batch_norm:
        obs = tf_old.contrib.layers.batch_norm(
            obs, is_training=training, scale=True, reuse=tf.get_variable_scope().reuse,
            epsilon=obs_norm_epsilon, decay=obs_norm_momentum, scope='obs_batch_norm'
        )

    return obs

