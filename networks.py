'''
Author:         David Radke
Affiliation:    University of Waterloo
Date:           May 2, 2022
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls
import os
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import utils
import pickle

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def imshow_zero_center(image, **kwargs):
    lim = tf.reduce_max(abs(image))
    plt.imshow(image, vmin=-lim, vmax=lim, cmap='seismic', **kwargs)
    plt.colorbar()
    plt.show()


class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN:
    def __init__(self, type, id, parameters, hidden_units, dir):
        action_amt = parameters.nactions
        self._dir = dir
        self._id = id
        self.nactions = action_amt
        self.batch_size = parameters.batch_size
        self.optimizer = tf.optimizers.Adam(parameters.learning_rate)
        self.gamma = parameters.gamma
        if parameters.load_model:
            print("Loading ", type, " model for agent: ", id)
            if type == 'train' and parameters.load_model:
                self.model = keras.models.load_model(self._dir+id+"-trainNet")
            else:
                self.model = keras.models.load_model(self._dir+id+"-targetNet")
        else:
            self.model = MyModel(parameters.nteams, hidden_units, action_amt)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = parameters.replay_buffer_max_length
        self.min_experiences = parameters.batch_size

        self._replace_random = parameters.replace_random
        self._replace_similar = parameters.replace_similar

    def saveExperience(self):
        dir = self._dir + "experience/"
        _ = utils.checkPath(dir)

        with open(dir+str(epi)+'experience.pkl', 'wb') as fp:
            pickle.dump(self.experience, fp, protocol=pickle.HIGHEST_PROTOCOL)


    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet, agent):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.nactions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon, team_q=None):
        q_vals = self.predict(np.atleast_2d(states))[0].numpy()

        if team_q is not None:
            for idx in range(q_vals.size):
                team_q[idx][int(self._id[-1])] = q_vals[idx]
            q = team_q[team_q > 0]
            q_vals = np.mean(team_q, axis=1)

        if np.random.random() < epsilon:
            return np.random.choice(self.nactions), q_vals
        else:
            return np.argmax(q_vals), q_vals

    def idxToReplace(self, exp, db=None):
        if self._replace_random:
            idx = np.random.randint(low=0, high=len(self.experience['s']))
        elif self._replace_similar:

            try:
                state_idxs = np.argwhere(np.argmax(np.array(self.experience['s']), axis=1) == np.argmax(exp['s']))
                action_idxs = np.argwhere(np.array(self.experience['a'])[state_idxs].flatten() == exp['a'])
                min = np.argmin(np.absolute(np.array(self.experience['r'])[state_idxs[action_idxs]] - exp['r']))
                idx = state_idxs[action_idxs][min][0][0]
            except:
                state_idxs = state_idxs.flatten()
                idx = state_idxs[np.random.randint(low=0, high=state_idxs.size)]

        return idx


    def add_experience(self, exp):
        ''' replace random and replace similar are replay buffer replacement methods I experimented with but did not include in any paper '''
        
        if self._replace_random or self._replace_similar:
            if len(self.experience['s']) >= self.max_experiences:
                idx = self.idxToReplace(exp)
                for key, value in exp.items():
                    # self.experience[key].pop(0)
                    self.experience[key][idx] = value
            else:
                for key, value in exp.items():
                    self.experience[key].append(value)
        else:
            if len(self.experience['s']) >= self.max_experiences:
                for key in self.experience.keys():
                    self.experience[key].pop(0)
            for key, value in exp.items():
                self.experience[key].append(value)


    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
