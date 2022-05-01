import pandas as pd
import json
import shutil,os
import numpy as np
import random
from joblib import Parallel, delayed
import multiprocessing

from constants import *
from utility import *

def get_preprocessed_data(styles, batch_size, time_steps):
    #load in metadata
    js = pd.read_json('../data/maestro-v3.0.0/maestro-v3.0.0.json')

    #list all composers, comp-genre pairs
    composers = []
    comp_genre = dict()
    for i in styles:
        for j in i:
            genre = j.split('/')[-2]
            comp = j.split('/')[-1]
            comp_genre[comp] = genre
            composers += [comp]

    #find all paths corresponding to a composer  
    def find_composer_path(df, name):
        
        return '../data/maestro-v3.0.0/' + df[df.canonical_composer.str.lower().str.contains(name)].midi_filename

    #composer with paths in a dictionary
    name_path = dict()
    for i in composers:
        name_path[i] = find_composer_path(js, i)

    for i in composers:
        paths = name_path[i]
        for j in paths:
            directory = os.path.join('../data', comp_genre[i], i)
            if not os.path.exists(directory):
                os.makedirs(directory)
            shutil.copy(j, directory)

    def load_all(styles, batch_size, time_steps):
        """
        Loads all MIDI files as a piano roll.
        (For Keras)
        """
        note_data = []
        beat_data = []
        style_data = []

        note_target = []

        styles = [y for x in styles for y in x]

        for style_id, style in enumerate(styles):
            style_hot = one_hot(style_id, NUM_STYLES)
            # Parallel process all files into a list of music sequences
            all_files = [os.path.join(style, f) for f in os.listdir(style)]
            seqs = Parallel(n_jobs=multiprocessing.cpu_count(), backend='threading')(delayed(load_midi)(f) for f in all_files)

            for seq in seqs:
                if len(seq) >= time_steps:
                    # Clamp MIDI to note range
                    seq = clamp_midi(seq)
                    # Create training data and labels
                    train_data, label_data = stagger(seq, time_steps)
                    note_data += train_data
                    note_target += label_data

                    beats = [compute_beat(i, NOTES_PER_BAR) for i in range(len(seq))]
                    beat_data += stagger(beats, time_steps)[0]

                    style_data += stagger([style_hot for i in range(len(seq))], time_steps)[0]

        note_data = np.array(note_data)
        beat_data = np.array(beat_data)
        style_data = np.array(style_data)
        note_target = np.array(note_target)
        return [note_data, note_target, beat_data, style_data], [note_target]

    train_data, train_labels = load_all(styles, BATCH_SIZE, SEQ_LEN)

    np.save("../data/note_data.npy", train_data[0])
    np.save("../data/note_target.npy", train_data[1])
    np.save("../data/beat_data.npy", train_data[2])
    np.save("../data/style_data.npy", train_data[3])

    return train_data, train_labels