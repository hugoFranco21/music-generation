import collections
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
import random

from tensorflow import keras
from matplotlib import pyplot as plt
from preprocessing import notes_to_midi, midi_to_notes, get_notes_from_files

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000
data_dir = pathlib.Path('dataset/')

filenames = glob.glob(str(data_dir/'*.mid*'))
print('Number of files:', len(filenames))

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)

def create_sequences(
    dataset: tf.data.Dataset, 
    seq_length: int,
    vocab_size = 128,
) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_length = seq_length+1
    key_order = ['pitch', 'step', 'duration']
    # Take 1 extra for the labels
    windows = dataset.window(seq_length, shift=1, stride=1,
                            drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    def scale_pitch(x):
        x = x/[vocab_size,1.0,1.0]
        return x

    # Split the labels
    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

def get_model(path, learning_rate):
    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = keras.models.load_model(path,compile=False)

    model.compile(loss=loss,
        loss_weights={
            'pitch': 0.05,
            'step': 1.0,
            'duration':1.0,
        },
        optimizer=optimizer,)
    return model

def main():
    seq_length = 25
    vocab_size = 128

    input_shape = (seq_length, 3)
    learning_rate = 0.001

    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,  return_sequences=True))(inputs)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(x)

    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(y),
        'step': tf.keras.layers.Dense(1, name='step')(y),
        'duration': tf.keras.layers.Dense(1, name='duration')(y),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=loss,
        loss_weights={
            'pitch': 0.05,
            'step': 1.0,
            'duration':1.0,
        },
        optimizer=optimizer,)

    print(model.summary())

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training-checkpoints/ckpt_{epoch}',
            monitor='loss',
            save_weights_only=True,),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True),
    ]

    epochs = 30

    model = get_model('model/note_rnn', learning_rate)

    i = 45000
    while i < 45129:
        notes, n = get_notes_from_files(filenames[i % 45129:i+10])
        i += 10
        print(i)
        seq_ds = create_sequences(notes, seq_length, vocab_size)

        batch_size = 64
        buffer_size = n - seq_length  # the number of items in the dataset
        train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

        history = model.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
        )

        model.save('model/note_rnn')

    plt.plot(history.epoch, history.history['loss'], label='total loss')
    plt.savefig('resources/loss.png')

if __name__ == '__main__':
    main()


"""
def predict_next_note(
    notes: np.ndarray, 
    keras_model: tf.keras.Model, 
    temperature: float = 1.0) -> int:
    #Generates a note IDs using a trained sequence model.

    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)

temperature = 2.0
num_predictions = 120

raw_notes = midi_to_notes(filenames[random.randint(0, 45128)])

sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

# The initial sequence of notes; pitch is normalized similar to training
# sequences
input_notes = (
    sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

generated_notes = []
prev_start = 0
for _ in range(num_predictions):
    pitch, step, duration = predict_next_note(input_notes, model, temperature)
    start = prev_start + step
    end = start + duration
    input_note = (pitch, step, duration)
    generated_notes.append((*input_note, start, end))
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    prev_start = start

generated_notes = pd.DataFrame(
    generated_notes, columns=(*key_order, 'start', 'end'))

out_file = 'output.midi'
out_pm = notes_to_midi(
    generated_notes, out_file=out_file, instrument_name='Electric Guitar (clean)')
    """