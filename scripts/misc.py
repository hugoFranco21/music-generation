import collections
import glob
import numpy as np
import pathlib
import pandas as pd
import tensorflow as tf
import random

from tensorflow import keras
from matplotlib import pyplot as plt
from preprocessing import notes_to_midi, midi_to_notes, get_notes_from_files
from training import mse_with_positive_pressure

loss = {
    'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
    'step': mse_with_positive_pressure,
    'duration': mse_with_positive_pressure,
}

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model = keras.models.load_model('model/note_rnn',compile=False)
"""model.compile(loss=loss,
    loss_weights={
        'pitch': 0.05,
        'step': 1.0,
        'duration':1.0,
    },
    optimizer=optimizer,)"""

print(model.get_weights())

print(model.summary())

