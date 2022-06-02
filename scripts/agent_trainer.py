import gym
import tensorflow as tf
from collections import deque

import random
import numpy as np
import math

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import History

MAX_EPSILON = 1
MIN_EPSILON = 0.01

GAMMA = 0.95
LAMBDA = 0.0005
TAU = 0.08

BATCH_SIZE = 32
REWARD_STD = 1.0

enviroment = gym.make("CartPole-v0")

NUM_STATES = 4
NUM_ACTIONS = enviroment.action_space.n

class ExpirienceReplay:
    def __init__(self, maxlen = 2000):
        self._buffer = deque(maxlen=maxlen)
    
    def store(self, state, action, reward, next_state, terminated):
        self._buffer.append((state, action, reward, next_state, terminated))
        
    def get_batch(self, batch_size):
        if no_samples > len(self._samples):
            return random.sample(self._buffer, len(self._samples))
        else:
            return random.sample(self._buffer, batch_size)
        
    def get_arrays_from_batch(self, batch):
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([(np.zeros(NUM_STATES) if x[3] is None else x[3]) 
                                for x in batch])
        
        return states, actions, rewards, next_states
        
    @property
    def buffer_size(self):
        return len(self._buffer)

