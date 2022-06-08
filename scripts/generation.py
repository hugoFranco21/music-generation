import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
import random

from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple
from preprocessing import midi_to_notes, notes_to_midi
from training import get_model

model = get_model('model/note_rnn', 0.001)

def predict_next_note(
    notes: np.ndarray, 
    keras_model: tf.keras.Model, 
    temperature: float = 1.0) -> int:
    """Generates a note IDs using a trained sequence model."""

    print(notes.shape)
    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = keras_model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    soft = tf.nn.softmax(pitch_logits)
    pitch = tf.math.argmax(soft, axis=-1)
    probability = tf.math.reduce_max(soft)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(duration, 0)

    return int(pitch), float(step), float(duration), float(probability)

def main():
    num_predictions = 1

    data_dir = pathlib.Path('dataset/')

    filenames = glob.glob(str(data_dir/'*.mid*'))
    seq_length = 25
    vocab_size = 128
    file = filenames[random.randint(0, 45128)]
    print(file)
    raw_notes = midi_to_notes('dataset/0a0ce238fb8c672549f77f3b692ebf32.mid')
    key_order = ['pitch', 'step', 'duration']
    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    # The initial sequence of notes; pitch is normalized similar to training
    # sequences
    input_notes = (
        sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    #print('input notes')
    #print(input_notes)

    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        pitch, step, duration, _ = predict_next_note(input_notes, model)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        print(f'input note {input_note}')
        generated_notes.append((*input_note, start, end))
        #print('note: ')
        #print(input_note)
        #print(start)
        #print(end)
        input_notes = np.delete(input_notes, 0, axis=0)
        print('input_notes after delete')
        print(input_notes)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        #print('input_notes after append')
        #print(input_notes)
        prev_start = start

    #print(generated_notes)

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*key_order, 'start', 'end'))

    out_file = 'output.midi'
    out_pm = notes_to_midi(
        generated_notes, out_file=out_file, instrument_name='Electric Guitar (Clean)')

if __name__ == '__main__':
    main()