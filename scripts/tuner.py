import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
import glob
import pathlib
from training import get_model

import random

EPISODES = 500
TRAIN_END = 0
REWARD_SCALER = 1.0
BAR_LENGTH = 4

FIRST = 0
SECOND = 2
THIRD = 4
FOURTH = 5
FIFTH = 7
SIXTH = 9
SEVENTH = 11
EIGHTH = 12
SCALE = [FIRST,SECOND,THIRD,FOURTH,FIFTH,SIXTH,SEVENTH,EIGHTH]
TWELVE_FORM = [FIRST, FOURTH, FIRST ,FIRST ,FOURTH,FOURTH, FIRST,FIRST ,FIFTH,FOURTH,FIRST,FIFTH]
EIGHT_FORM = [FIRST,FIRST,FIRST,FIRST,FOURTH,FOURTH,FIFTH,FIRST]
SIXTEEN_FORM = [FIRST,FIRST,FIRST,FIRST,FIRST,FIRST,FIRST,FIRST,FOURTH,FOURTH,FIRST,FIRST,FIFTH,FOURTH,FIRST,FIRST]

REWARD_RNN = tf.keras.load_model('model/note_rnn', compile=False)

def discount_rate(): #Gamma
    return 0.95

def learning_rate(): #Alpha
    return 0.001

def batch_size():
    return 24

def next_note_probabilities_rnn(
    notes: np.ndarray, 
    a: np.ndarray,
    keras_model: tf.keras.Model = REWARD_RNN,
    ):
    inputs = tf.expand_dims(notes, 0)
    inputs = inputs[:25]
    predictions = keras_model.predict(inputs)
    pitch_logits = predictions['pitch']
    
    soft = tf.nn.softmax(pitch_logits)

    probability = soft[a[0]]

    return float(probability)

def predict_next_note(
    notes: np.ndarray, 
    keras_model: tf.keras.Model, 
    temperature: float = 1.0) -> int:
    """Generates a note IDs using a trained sequence model."""

    assert temperature > 0

    print(notes.shape)
    # Add batch dimension
    inputs = inputs[:25]
    inputs = tf.expand_dims(notes, 0)

    predictions = keras_model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    soft = tf.nn.softmax(pitch_logits)
    pitch = tf.math.argmax(soft)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(duration, 0)

    return int(pitch), float(step), float(duration)

class DoubleDeepQNetwork():
    def __init__(self, states, actions, alpha, gamma, epsilon,epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.gamma = gamma
        self.composition = []
        self.beat = 0
        self.input = []
        #Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model(self, 'model/note_rnn')
        self.model_target = self.build_model(self, 'model/note_rnn') #Second (target) neural network
        self.update_target_from_model() #Update weights
        self.loss = []

    def build_model(self, path):
        model = tf.keras.models.load_model(path,compile=False)
        model.compile(loss=self.loss(self), #Loss function: Mean Squared Error
                    optimizer=tf.keras.optimizers.Adam(lr=self.alpha))
        return model

    def loss(self):
        def music_rewards_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
            loss = (tf.math.log(next_note_probabilities_rnn(self.composition[-26:], y_pred))+ (1/REWARD_SCALER)*self.get_rewards(self, y_pred) + 
                self.gamma*next_note_probabilities_rnn(self.composition[-26:], y_pred, self.model_target) - 
                next_note_probabilities_rnn(self.composition[-26:], y_pred, self.model)) ** 2
            return tf.reduce_mean(loss)
        return music_rewards_loss

    def get_rewards(self, action):
        reward = 0
        reward += self.reward_key(self,action,)
        reward += self.penalize_repeating(self,action,-100.0)
        reward_rock = self.reward_twelve_bar(self,action)
        if reward_rock < 0:
            reward_rock += self.reward_eight_bar(self,action,)
        if reward_rock < 0:
            reward_rock += self.reward_sixteen_bar(self,action)
        reward += reward_rock
        reward += self.penalize_invalid_duration(self,action)
        reward += self.penalize_invalid_step(self, action)
        return reward

    def reward_key(self, action, amount=-1):
        if not (action[0] - self.key) in SCALE:
            reward = amount
        return reward

    def penalize_repeating(self, action, amount=-100.0):
        num_repeated = 0
        for i in range(len(self.composition)-1, -1, -1):
            if self.composition[i][0] == action[0]:
                num_repeated += 1
        if num_repeated > 8:
            return amount
        return 0.0
    
    def reward_twelve_bar(self,action,amount=15.0):
        length = len(self.composition)
        if (length > BAR_LENGTH):
            if(length % BAR_LENGTH == 1):
                bar = length / BAR_LENGTH
                index = (bar - 1) % (len(TWELVE_FORM))
                if (action[0] - self.key) == TWELVE_FORM[index]:
                    return amount
                else:
                    return -1.0
            else:
                return 0.0
        else:
            return 0.0

    def reward_eight_bar(self,action,amount=15.0):
        length = len(self.composition)
        if (length > BAR_LENGTH):
            if(length % BAR_LENGTH == 1):
                bar = length / BAR_LENGTH
                index = (bar - 1) % (len(EIGHT_FORM))
                if (action[0] - self.key) == EIGHT_FORM[index]:
                    return amount
                else:
                    return 0.0
        else:
            return 0.0

    def reward_sixteen_bar(self,action,amount=15.0):
        length = len(self.composition)
        if (length > BAR_LENGTH):
            if(length % BAR_LENGTH == 1):
                bar = length / BAR_LENGTH
                index = (bar - 1) % (len(SIXTEEN_FORM))
                if (action[0] - self.key) == SIXTEEN_FORM[index]:
                    return amount
                else:
                    return -1.0
            else:
                return 0.0
        else:
            return 0.0

    def penalize_invalid_duration(self, action):
        if action[2] < 0.1:
            return -50.0
        if action[2] > 3.5:
            return -50.0
        return 1.0

    def penalize_invalid_step(self, action):
        if action[1] < 0.1:
            return -50.0
        if action[1] > 2.5:
            return -50.0
        return 1.0

    def prime_models():
        data_dir = pathlib.Path('dataset/')

        filenames = glob.glob(str(data_dir/'*.mid*'))
        seq_length = 25
        vocab_size = 128
        file = filenames[random.randint(0, 45128)]
        print(file)
        raw_notes = midi_to_notes('dataset/0a0ce238fb8c672549f77f3b692ebf32.mid')
        key_order = ['pitch', 'step', 'duration']
        sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    def update_target_from_model(self):
        #Update the target model from the base model
        self.model_target.set_weights( self.model.get_weights() )

    def action(self, state):
        pitch, step, duration = predict_next_note(self.input, self.model)
        if len(self.composition == 0):
            prev_start = 0
        else:
            prev_start = self.composition[-1:][3]
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        self.composition.append((*input_note, start, end))
        self.input = np.delete(self.input, 0, axis=0)
        self.input = np.append(self.input, np.expand_dims(input_note, 0), axis=0)
        return [pitch, step, duration]

    def store(self, state, action, reward, nstate, done):
        #Store the experience in memory
        self.memory.append( (state, action, reward, nstate, done) )

    def experience_replay(self, batch_size):
        #Execute the experience replay
        minibatch = random.sample( self.memory, batch_size ) #Randomly sample from memory

        #Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = np.array(minibatch)
        st = np.zeros((0,self.nS)) #States
        nst = np.zeros( (0,self.nS) )#Next States
        for i in range(len(np_array)): #Creating the state and next state np arrays
            st = np.append( st, np_array[i,0], axis=0)
            nst = np.append( nst, np_array[i,3], axis=0)
        st_predict = self.model.predict(st) #Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        nst_predict_target = self.model_target.predict(nst) #Predict from the TARGET
        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            #Predict from state
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]
            if done == True: #Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:   #Non terminal
                target = reward + self.gamma * nst_action_predict_target[np.argmax(nst_action_predict_model)] #Using Q to get T is Double DQN
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        #Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size,self.nS)
        y_reshape = np.array(y)
        epoch_count = 1
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        #Graph Losses
        for i in range(epoch_count):
            self.loss.append( hist.history['loss'][i] )
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay