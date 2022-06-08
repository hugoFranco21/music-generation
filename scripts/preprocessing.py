import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
import random
import collections

from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

def notes_to_midi(
    notes: pd.DataFrame,
    out_file: str, 
    instrument_name: str,
    velocity: int = 100,  # note loudness
    dir='output/'
)-> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
        instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(dir + out_file)
    return pm

def midi_to_notes(midi_file: str, instrument_index = 0) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[instrument_index]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def get_notes_from_files(files: list) -> tf.data.Dataset:
    all_notes = []
    for f in files:
        pm = None
        instruments = None
        try:
            pm = pretty_midi.PrettyMIDI(f)
            instruments = pm.instruments
        except:
            continue
        index = 0
        i = 0
        for instrument in instruments:
            instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
            if 'guitar' in instrument_name.lower():
                index = i
                break
            i += 1
        i = 0
        if index == 0:
            for instrument in instruments:
                instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
                if 'piano' in instrument_name.lower():
                    index = i
                    break
                i += 1
        all_notes.append(midi_to_notes(f, index))
    all_notes = pd.concat(all_notes)
    n_notes = len(all_notes)
    key_order = ['pitch', 'step', 'duration']
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    return notes_ds, n_notes, train_notes