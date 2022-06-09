import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
import glob
import pathlib
from training import mse_with_positive_pressure, create_sequences
from preprocessing import midi_to_notes, get_notes_from_files
import helpers
import pandas as pd

import random

EPISODES = 500
TRAIN_END = 0
REWARD_SCALER = 1.0
REWARD_RNN = tf.keras.models.load_model('model/note_rnn', compile=False)

def discount_rate():  # Gamma
    return 0.95

def learning_rate():  # Alpha
    return 0.001

def batch_size():
    return 24

def next_note_probabilities_rnn(
    notes: np.ndarray,
    a: np.ndarray,
    keras_model: tf.keras.Model = REWARD_RNN,
):
    probability = 0 
    for note in a:
        inputs = tf.expand_dims(notes, 0)
        predictions = keras_model.predict(inputs)
        pitch_logits = predictions['pitch']
        step = predictions['step']
        duration = predictions['duration']
        step = tf.maximum(0, step)
        duration = tf.maximum(duration, 0)
        soft = tf.nn.softmax(pitch_logits)
        soft = np.squeeze(soft)
        pitch = tf.math.argmax(soft, axis=-1)
        probability += soft[note[0]]
        input_note = (note[0], note[1], note[2])
        notes = np.delete(notes, 0, axis=0)
        notes = np.append(notes, np.expand_dims(input_note, 0), axis=0)

    return float(probability / len(a)), float(step), float(duration)

def predict_next_note(
        notes: np.ndarray,
        keras_model: tf.keras.Model,
        temperature: float = 1.0) -> int:
    """Generates a note IDs using a trained sequence model."""

    assert temperature > 0

    # Add batch dimension
    #print(notes.shape)
    inputs = tf.expand_dims(notes, 0)

    predictions = keras_model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    soft = tf.nn.softmax(pitch_logits)
    pitch = tf.math.argmax(soft, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)


    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(duration, 0)

    return int(pitch), float(step), float(duration)

class DoubleDeepQNetwork():
    def __init__(self, initialize_from_hd, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.memory = deque([], maxlen=10000)
        self.alpha = alpha
        self.gamma = gamma
        self.composition = []
        self.beat = 0
        self.num_times_stored_called = 0
        self.minibatch_size = 24
        self.reward = 0
        self.epochs = 0
        self.epochs_array = []
        # Explore/Exploit
        self.epsilon = epsilon
            
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        if not initialize_from_hd:
            self.model = self.build_model('model/note_rnn')
            self.model_target = self.build_model(
                'model/note_rnn')  # Second (target) neural network
        else:
            self.model = self.build_model('model/q_net')
            self.model_target = self.build_model(
                'model/q_net')
        self.prime_models()
        self.update_target_from_model()  # Update weights
        self.loss = []

    def build_model(self, path):
        
        model = tf.keras.models.load_model(path, compile=False)
        loss = {
            'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            'step': mse_with_positive_pressure,
            'duration': mse_with_positive_pressure,
        }

        optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha)

        model.compile(loss=loss,
                        loss_weights = {
                            'pitch': 0.1,
                            'step': 0.05,
                            'duration':0.05,
                        },
                        optimizer=optimizer)
        return model

    def get_total_reward(self):
        return self.reward

    def get_complete_reward(self, action):
        sum = 0.0
        prob, _, _ = next_note_probabilities_rnn(self.input,
                        [action])
        sum += tf.math.log(prob)
        sum += self.get_rewards(action)
        return sum

    def get_rewards(self, action):
        reward = 0
        reward += self.reward_key(action)
        reward += self.penalize_repeating(action, -100.0)
        reward_rock = self.reward_twelve_bar(action)
        if reward_rock < 0:
            reward_rock += self.reward_eight_bar(action,)
        if reward_rock < 0:
            reward_rock += self.reward_sixteen_bar(action)
        reward += reward_rock
        reward += self.penalize_invalid_duration(action)
        reward += self.penalize_invalid_step(action)
        return reward

    def reward_key(self, action, amount=-1):
        reward = 10
        if not ((action[0] - self.key) % 12) in helpers.SCALE:
            reward = amount
        return reward

    def penalize_repeating(self, action, amount=-100.0):
        num_repeated = 0
        for i in range(len(self.composition)-1, -1, -1):
            if self.composition[i][0] == action[0]:
                num_repeated += 1
        if num_repeated > 12:
            return amount
        return 0.0

    def reward_twelve_bar(self, action, amount=15.0):
        length = len(self.composition)
        if (length > helpers.BAR_LENGTH):
            if(length % helpers.BAR_LENGTH == 1):
                bar = length // helpers.BAR_LENGTH
                index = (bar - 1) % (len(helpers.TWELVE_FORM))
                if (action[0] - self.key) == helpers.TWELVE_FORM[index]:
                    return amount
                else:
                    return -1.0
            else:
                return 0.0
        else:
            return 0.0

    def reward_eight_bar(self, action, amount=15.0):
        length = len(self.composition)
        if (length > helpers.BAR_LENGTH):
            if(length % helpers.BAR_LENGTH == 1):
                bar = length // helpers.BAR_LENGTH
                index = (bar - 1) % (len(helpers.EIGHT_FORM))
                if (action[0] - self.key) == helpers.EIGHT_FORM[index]:
                    return amount
                else:
                    return -1.0
            else:
                return 0.0
        else:
            return 0.0

    def reward_sixteen_bar(self, action, amount=15.0):
        length = len(self.composition)
        if (length > helpers.BAR_LENGTH):
            if(length % helpers.BAR_LENGTH == 1):
                bar = length // helpers.BAR_LENGTH
                index = (bar - 1) % (len(helpers.SIXTEEN_FORM))
                if (action[0] - self.key) == helpers.SIXTEEN_FORM[index]:
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
        if action[2] > 10:
            return -100
        if action[2] > 3.5:
            return -50.0
        return 1.0

    def penalize_invalid_step(self, action):
        if action[1] < 0.1:
            return -50.0
        if action[1] > 10:
            return -100
        if action[1] > 2.5:
            return -50.0
        return 1.0

    def prime_models(self):
        data_dir = pathlib.Path('dataset/')

        filenames = glob.glob(str(data_dir/'*.mid*'))
        seq_length = 25
        vocab_size = 128
        file = filenames[random.randint(0,500)]
        raw_notes = midi_to_notes(file)
        key_order = ['pitch', 'step', 'duration']
        sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
        input_notes = (
            sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))
        self.input = input_notes
        self.key = sample_notes[0][0]
        return input_notes

    def update_target_from_model(self):
        # Update the target model from the base model
        self.model_target.set_weights(self.model.get_weights())

    def action(self, model):
        pitch, step, duration = predict_next_note(self.input, model)
        if len(self.composition) == 0:
            prev_start = 0
        else:
            prev_start = self.composition[-1][3]
        start = prev_start + step
        end = start + duration
        state = self.input
        input_note = (pitch, step, duration)
        self.composition.append((*input_note, start, end))
        self.beat += 1
        self.input = np.delete(self.input, 0, axis=0)
        self.input = np.append(
            self.input, np.expand_dims(input_note, 0), axis=0)
        action = [pitch, step, duration]
        reward = self.get_complete_reward(action)
        self.reward += reward
        return state, action, reward, self.input

    def reset_composition(self):
        """Starts the models internal composition over at beat 0, with no notes.
        Also resets statistics about whether the composition is in the middle of a
        melodic leap.
        """
        self.beat = 0
        self.reward = 0
        self.composition = []

    def store(self, state, action, reward, nstate):
        self.memory.append( (state, action, reward, nstate) )

    def experience_replay(self, batch_size):
        #Execute the experience replay
        minibatch = random.sample( self.memory, batch_size ) #Randomly sample from memory
        #Convert to numpy for speed by vectorization
        x = []
        y_pitch = []
        y_step = []
        y_duration =[]
        np_array = np.array(minibatch)
        st = np.zeros((batch_size, len(self.input), 3)) #States
        nst = np.zeros((batch_size, len(self.input), 3)) #Next States
        for i in range(len(np_array)): #Creating the state and next state np arrays
            st = np.append( st, np.expand_dims(np_array[i][0], 0), axis=0)
            nst = np.append( nst, np.expand_dims(np_array[i][3],0), axis=0)
        st_predict = self.model.predict(st) #Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        nst_predict_target = self.model_target.predict(nst) #Predict from the TARGET
        index = 0
        for state, action, reward, nstate in minibatch:
            x.append(state)
            #Predict note from state
            nst_action_predict_model_pitch = nst_predict['pitch'][index] # q prediction for pitch
            nst_action_predict_model_step = nst_predict['step'][index] # q prediction for step
            nst_action_predict_model_duration = nst_predict['duration'][index] # q prediction for duration
            nst_action_predict_target_pitch = nst_predict['pitch'][index] # target prediction for pitch
            nst_action_predict_target_step = nst_predict['step'][index] # target prediction for step
            nst_action_predict_target_duration = nst_predict['duration'][index] # target prediction for duration
            soft_model = tf.nn.softmax(nst_action_predict_model_pitch)
            soft_model = np.squeeze(soft_model)
            pitch_model = tf.math.argmax(soft_model, axis=-1)
            step_model = nst_action_predict_model_step
            duration_model = nst_action_predict_model_duration
            prob, step, duration = next_note_probabilities_rnn(state, [[pitch_model, step_model, duration_model]], self.model_target)
            target_pitch = reward + self.gamma * prob #Using Q to get T is Double DQN
            target_step = reward*0.1 + self.gamma * step
            target_duration = reward*0.1 + self.gamma * duration
            target_f = {}
            target_f['pitch'] = st_predict['pitch'][index]
            target_f['step'] = st_predict['step'][index]
            target_f['duration'] = st_predict['duration'][index] 
            target_f['pitch'][action[0]] = target_pitch
            target_f['step'] = target_step
            target_f['duration'] = target_duration
            soft_target = tf.nn.softmax(target_f['pitch'])
            soft_target = np.squeeze(soft_target)
            pitch_target = tf.math.argmax(soft_target, axis=-1)
            target_ds = [pitch_target, target_f['step'], target_f['duration']]
            y_pitch.append(pitch_target)
            y_step.append(target_f['step'])
            y_duration.append(target_f['duration'])
            index += 1
        #Reshape for Keras Fit
        x_reshape = np.array(x, dtype=np.float32).reshape((batch_size,len(self.input), 3))
        #x_ds = np.asarray(x_reshape).astype(np.float32)
        #y_reshape = np.array(y).reshape(batch_size, (128, 1,1), 3 )
        #y_ds = np.asarray(y).astype(np.float32)
        y_pitch_res = np.array(y_pitch, dtype=np.float32)
        y_step_res = np.array(y_step, dtype=np.float32)
        y_duration_res = np.array(y_duration, dtype=np.float32)
        #print(f'y_reshape {y_reshape.shape}')
        #print(f'example {y_reshape[0]}')
        #y_reshape = np.expand_dims(y_reshape,1)
        #print(f'example {y_reshape}')
        epoch_count = 1
        hist = self.model.fit(x_reshape, {'pitch': y_pitch_res, 'step': y_step_res, 'duration': y_duration_res}, epochs=epoch_count, verbose=1)
        #Graph Losses
        for i in range(epoch_count):
            self.loss.append( hist.history['loss'][i] )
        self.epochs += 1
        self.epochs_array.append(self.epochs)
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

dqn = DoubleDeepQNetwork(False, learning_rate(), discount_rate(), 1, 0.001, 0.995 )
print(dqn.model.summary())
batch_size = batch_size()

for e in range(EPISODES):
    print(f'iter {e}:')
    dqn.reset_composition()
    state = dqn.prime_models()
    if len(state) < 25:
        continue
    #print(state)
    state = np.reshape(state, (25,3)) # Resize to store in memory to pass to .predict
    tot_rewards = 0
    for i in range(72): #72 is the length of a new composition we want
        state, action, reward, nstate = dqn.action(dqn.model)
        #nstate = np.reshape(nstate, [1, nS])
        tot_rewards += reward
        dqn.store(state, action, reward, nstate) # Resize to store in memory to pass to .predict
        state = nstate
        #Experience Replay
        if len(dqn.memory) > 0 and len(dqn.memory) % batch_size == 0:
            dqn.experience_replay(batch_size)
    #Update the weights after each episode (You can configure this for x steps as well
    dqn.update_target_from_model()

    plt.plot(dqn.epochs_array, dqn.loss, label='total loss')
    plt.savefig('resources/rl_loss.png')
    dqn.model.save('model/tuner')


    