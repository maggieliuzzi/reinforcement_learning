"""
Double Deep Q-Learning with Prioritised Experience Replay (DDQN)
"""

import numpy as np
import tensorflow.compat.v1 as tf
from agents.prioritised_replay import PrioritisedReplayBuffer


class DDQN_PrioritisedReplay:
    def __init__(
            self,
            n_actions: int,
            n_features: int,
            learning_rate: float = 0.01,
            reward_decay: float = 0.9,
            e_greedy: float = 0.9,
            replace_target_iter: int = 300,
            memory_size: int = 500,
            batch_size: int = 32,
            e_greedy_increment=None
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0

        self.memory = PrioritisedReplayBuffer(n_features)  #n_features * 2 + 2   # np.zeros((self.memory_size, n_features * 2 + 2))  # shape: (2000, 8)

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        self.cost_his = []

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_target')  # for calculating loss
        
        with tf.variable_scope('eval_net', reuse=None):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                 tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            with tf.variable_scope('l1', reuse=None):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2', reuse=None):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss', reuse=None):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        
        with tf.variable_scope('train', reuse=None):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net',reuse=None):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1',reuse=None):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2',reuse=None):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]  # [[1 0 0]] shape: (1, 3)

        if np.random.uniform() < self.epsilon:
            # observation: eg. [[0.61752891 0.71073113 0.33692625]]
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})  # using evaluation network to get Q(s,a) for each a, from curr state
            # eg. [[-0.34863967  0.03011987  0.27612218  0.21124053  0.22055164  0.23672578 -0.13390371 -0.26629397  0.23533061]]
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _getTargets(self, batch):
        no_state = np.zeros(self.n_features)

        states = np.array([ o[1][0:3] for o in batch ])  # s: [[1 0 0]]
        next_states = np.array([ (no_state if o[1][-3:] is None else o[1][-3:]) for o in batch ])  # st+1: [[ 0.99691733  0.         -0.0784591 ]]

        # actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        
        p = self.sess.run(self.q_eval, feed_dict={self.s: states})  # uses primary/evaluation network  # predicted values of actions
        p_next = self.sess.run(self.q_eval, feed_dict={self.s: next_states})  # uses primary/evaluation network
        p_next_target = self.sess.run(self.q_next, feed_dict={self.s_: next_states})  # uses target network

        x = np.zeros((len(batch), self.n_features))
        y = np.zeros((len(batch), self.n_actions))
        errors = np.zeros(len(batch))

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0:3]; a = o[3]; r = o[4]; s_ = o[-3:]  # s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            a = int(a)
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + self.gamma * p_next_target[i][ np.argmax(p_next[i]) ]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        index = self.memory_counter % self.memory_size
        sample = np.hstack((s, [a, r], s_))  # sample = (s, a, r, s_)

        x, y, errors = self._getTargets([(0, sample)])
        # Adding transition to memory
        self.memory.add(errors[0], sample)  # try adding transition instead  ## self.memory[index, :] = transition

        self.memory_counter += 1

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        '''
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        '''
        # Sampling transitions from memory
        batch = self.memory.sample(self.batch_size)
        x, y, errors = self._getTargets(batch)
        # update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        batch = np.array(batch)  # shape: (32, 2)
        batch = batch[:, 1]
        arr_curr = []
        arr_next = []
        actions = []
        rewards = []
        for i in batch:
            arr_curr.append(list(i[0:3]))
            arr_next.append(list(i[-3:]))
            actions.append(i[3].astype(int))  # self.n_features
            rewards.append(i[4])  # self.n_features + 1

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: arr_next,  # fixed params
                self.s: arr_curr  # newest params
            })

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = actions
        reward = rewards

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: arr_curr, self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
