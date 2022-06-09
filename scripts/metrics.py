import glob
import numpy as np
import pathlib
import pandas as pd
import tensorflow as tf
import random

from matplotlib import pyplot as plt
from preprocessing import midi_to_notes, notes_to_midi
from training import get_model
from generation import predict_next_note
import helpers

model = get_model('model/tuner', 0.001)

def  predict_next_note (
    notes: np.ndarray, 
    keras_model: tf.keras.Model, 
    temperature: float = 1.0) -> int:
    """Generates a note IDs using a trained sequence model."""

    # Add batch dimension
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
    pitch = tf.minimum(pitch, 127)
    pitch = tf.maximum(0,pitch)
    step = tf.maximum(0, step)
    duration = tf.maximum(duration, 0)
    return int(pitch), float(step), float(duration)


keys = []
repeating = []
twelve_bar = []
eight_bar = []
sixteen_bar = []
pitch = []
duration = []
step = []

def evaluate_composition(df: pd.DataFrame, key):
    keys.append(evaluate_key(df, key))
    repeating.append(evaluate_repeating(df))
    twelve_bar.append(evaluate_twelve_bar(df, key))
    eight_bar.append(evaluate_eigth_bar(df, key))
    sixteen_bar.append(evaluate_sixteen_bar(df, key))
    duration.append(evaluate_invalid_duration(df))
    step.append(evaluate_invalid_step(df))

def get_averages():
    return np.mean(np.array(keys)), \
        np.mean(np.array(repeating)), \
        np.mean(np.array(twelve_bar)), \
        np.mean(np.array(eight_bar)), \
        np.mean(np.array(sixteen_bar)), \
        np.mean(np.array(duration)), \
        np.mean(np.array(step))

def evaluate_key(df: pd.DataFrame, key):
    count = 0
    for i in range(0, len(df)):
        if ((df.iloc[i]['pitch'] - key) % 12) in helpers.SCALE:
            count += 1
    return float(count/len(df.index))
    
def evaluate_repeating(df: pd.DataFrame):
    arr = np.zeros(128)
    for i in range(0, len(df)):
        arr[int(df.iloc[i]['pitch'])] += 1
    return 1.0 if arr[np.argmax(arr)] >= 12 else 0.0

def evaluate_twelve_bar(df: pd.DataFrame, key):
    i = 1
    count = 0.0
    for j in range(0, len(df)):
        if i < helpers.BAR_LENGTH:
            i += 1
            continue
        i += 1
        if(i % helpers.BAR_LENGTH == 1):
            bar = i // helpers.BAR_LENGTH
            index = (bar) % (len(helpers.TWELVE_FORM))
            if (df.iloc[j]['pitch'] - key) == helpers.TWELVE_FORM[index]:
                count += 1.0/15.0
    return count

def evaluate_eigth_bar(df: pd.DataFrame, key):
    i = 1
    count = 0.0
    for j in range(0, len(df)):
        if i < helpers.BAR_LENGTH:
            i += 1
            continue
        i += 1
        if(i % helpers.BAR_LENGTH == 1):
            bar = i // helpers.BAR_LENGTH
            index = (bar - 1) % (len(helpers.EIGHT_FORM))
            if (df.iloc[j]['pitch'] - key) == helpers.EIGHT_FORM[index]:
                count += 1.0/15.0
    return count

def evaluate_sixteen_bar(df: pd.DataFrame, key):
    i = 1
    count = 0.0
    for j in range(0, len(df)):
        if i < helpers.BAR_LENGTH:
            i += 1
            continue
        i += 1
        if(i % helpers.BAR_LENGTH == 1):
            bar = i // helpers.BAR_LENGTH
            index = (bar - 1) % (len(helpers.SIXTEEN_FORM))
            if (df.iloc[j]['pitch'] - key) == helpers.SIXTEEN_FORM[index]:
                count += 1.0/15.0
    return count

def evaluate_invalid_duration(df: pd.DataFrame):
    count = 0
    for j in range(0, len(df)):
        if (df.iloc[j]['duration'] <= 0.1 or df.iloc[j]['duration'] >= 3.5):
            count += 1
    return float(count/len(df.index))

def evaluate_invalid_step(df: pd.DataFrame):
    count = 0
    for j in range(0, len(df)):
        if (df.iloc[j]['step'] <= 0.1 or df.iloc[j]['step'] >= 2.5):
            count += 1
    return float(count/len(df.index))

def write_to_file(f, 
            key_avg, 
            repeating_avg, 
            twelve_bar_avg, 
            eight_bar_avg, 
            sixteen_bar_avg,
            duration_avg, 
            step_avg):
    f.write("Average of notes in key: " + str(key_avg * 100) + " %\n")
    f.write("Compositions with repeated notes: " + str(repeating_avg * 100) + " %\n")
    f.write("Compositions that follow the eight bar rule: " + str(eight_bar_avg * 100) + " %\n")
    f.write("Compositions that follow the twelve bar rule: " + str(twelve_bar_avg * 100) + " %\n")
    f.write("Compositions that follow the sixteen bar rule: " + str(sixteen_bar_avg * 100) + " %\n")
    f.write("Average of notes with invalid duration: " + str(duration_avg * 100) + " %\n")
    f.write("Average of notes with invalid step: " + str(step_avg * 100) + " %\n\n")


num_predictions = 1000

def evaluate_model():
    data_dir = pathlib.Path('dataset/')

    filenames = glob.glob(str(data_dir/'*.mid*'))
    random.shuffle(filenames)
    seq_length = 25
    vocab_size = 128
    i = 0
    sem = random.randint(1000, 3000)
    filenames = filenames[sem:sem+1000]
    f = open("metrics/rl_rnn.txt", "a")
    while i < num_predictions:
        file = filenames[i]
        i += 1        
        raw_notes = midi_to_notes(file)
        key_order = ['pitch', 'step', 'duration']
        sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

        # The initial sequence of notes; pitch is normalized similar to training
        # sequences
        input_notes = (
            sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

        #print(input_notes)
        if len(input_notes) < 25:
            continue

        generated_notes = []
        composition_size = 64
        prev_start = 0
        key = 60
        for j in range(composition_size):
            pitch, step, duration = predict_next_note(input_notes, model)
            if j == 0:
                key = pitch
            start = prev_start + step
            end = start + duration
            input_note = (pitch, step, duration)
            generated_notes.append((*input_note, start, end))
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
            prev_start = start

        generated_notes = pd.DataFrame(
            generated_notes, columns=(*key_order, 'start', 'end'))
        
        evaluate_composition(generated_notes, key)
        
        if i % 20 == 0:
            key_avg, repeating_avg, twelve_bar_avg, eight_bar_avg, sixteen_bar_avg, duration_avg, step_avg  = get_averages()
            write_to_file(f, key_avg, repeating_avg, twelve_bar_avg, eight_bar_avg, sixteen_bar_avg, duration_avg, step_avg)

        if i % 75 == 0:
            out_file = f'output_{i}_rl.midi'
            out_pm = notes_to_midi(
                generated_notes, out_file=out_file, instrument_name='Electric Guitar (Clean)')
    f.close()

evaluate_model()