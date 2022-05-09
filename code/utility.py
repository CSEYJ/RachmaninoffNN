'''
Modified reimplementation of util.py (https://github.com/calclavia/DeepJ/blob/icsc/util.py) and midi_util.py (https://github.com/calclavia/DeepJ/blob/icsc/midi_util.py).

Credit to the original implementation:

MIT License

Copyright (c) 2018 Calclavia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import midi
import torch
import numpy as np
import os
import math
import random
from joblib import Parallel, delayed
import multiprocessing

from constants import *

# decode midi files to piano roll files
def midi_decode(pattern,
                classes=MIDI_MAX_NOTES,
                step=None):
    """
    Takes a MIDI pattern and decodes it into a piano roll.
    """
    if step is None:
        step = pattern.resolution // NOTES_PER_BEAT


    # Extract all tracks at highest resolution
    merged_replay = None
    merged_volume = None


    for track in pattern:
        # The downsampled sequences
        replay_sequence = []
        volume_sequence = []


        # Raw sequences
        replay_buffer = [np.zeros((classes,))]
        volume_buffer = [np.zeros((classes,))]


        for i, event in enumerate(track):
            # Duplicate the last note pattern to wait for next event
            for _ in range(event.tick):
                replay_buffer.append(np.zeros(classes))
                volume_buffer.append(np.copy(volume_buffer[-1]))


                # Buffer & downscale sequence
                if len(volume_buffer) > step:
                    # Take the min
                    replay_any = np.minimum(np.sum(replay_buffer[:-1], axis=0), 1)
                    replay_sequence.append(replay_any)


                    # Determine volume by max
                    volume_sum = np.amax(volume_buffer[:-1], axis=0)
                    volume_sequence.append(volume_sum)


                    # Keep the last one (discard things in the middle)
                    replay_buffer = replay_buffer[-1:]
                    volume_buffer = volume_buffer[-1:]


            if isinstance(event, midi.EndOfTrackEvent):
                break


            # Modify the last note pattern
            if isinstance(event, midi.NoteOnEvent):
                pitch, velocity = event.data
                volume_buffer[-1][pitch] = velocity / MAX_VELOCITY


                # Check for replay_buffer, which is true if the current note was previously played and needs to be replayed
                if len(volume_buffer) > 1 and volume_buffer[-2][pitch] > 0 and volume_buffer[-1][pitch] > 0:
                    replay_buffer[-1][pitch] = 1
                    # Override current volume with previous volume
                    volume_buffer[-1][pitch] = volume_buffer[-2][pitch]


            if isinstance(event, midi.NoteOffEvent):
                pitch, velocity = event.data
                volume_buffer[-1][pitch] = 0


        # Add the remaining
        replay_any = np.minimum(np.sum(replay_buffer, axis=0), 1)
        replay_sequence.append(replay_any)
        volume_sequence.append(volume_buffer[0])


        replay_sequence = np.array(replay_sequence)
        volume_sequence = np.array(volume_sequence)
        assert len(volume_sequence) == len(replay_sequence)


        if merged_volume is None:
            merged_replay = replay_sequence
            merged_volume = volume_sequence
        else:
            # Merge into a single track, padding with zeros of needed
            if len(volume_sequence) > len(merged_volume):
                # Swap variables such that merged_notes is always at least
                # as large as play_sequence
                tmp = replay_sequence
                replay_sequence = merged_replay
                merged_replay = tmp


                tmp = volume_sequence
                volume_sequence = merged_volume
                merged_volume = tmp


            assert len(merged_volume) >= len(volume_sequence)


            diff = len(merged_volume) - len(volume_sequence)
            merged_replay += np.pad(replay_sequence, ((0, diff), (0, 0)), 'constant')
            merged_volume += np.pad(volume_sequence, ((0, diff), (0, 0)), 'constant')


    merged = np.stack([np.ceil(merged_volume), merged_replay, merged_volume], axis=2)
    # Prevent stacking duplicate notes to exceed one.
    merged = np.minimum(merged, 1)
    return merged
    
def midi_encode(note_seq, resolution=NOTES_PER_BEAT, step=1):
    """
    Takes a piano roll and encodes it into MIDI pattern
    """
    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern()
    pattern.resolution = resolution
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)

    play = note_seq[:, :, 0]
    replay = note_seq[:, :, 1]
    volume = note_seq[:, :, 2]

    # The current pattern being played
    current = np.zeros_like(play[0])
    # Absolute tick of last event
    last_event_tick = 0
    # Amount of NOOP ticks
    noop_ticks = 0

    for tick, data in enumerate(play):
        data = np.array(data)

        if not np.array_equal(current, data):# or np.any(replay[tick]):
            noop_ticks = 0

            for index, next_volume in np.ndenumerate(data):
                if next_volume > 0 and current[index] == 0:
                    # Was off, but now turned on
                    evt = midi.NoteOnEvent(
                        tick=(tick - last_event_tick) * step,
                        velocity=int(volume[tick][index[0]] * MAX_VELOCITY),
                        pitch=index[0]
                    )
                    track.append(evt)
                    last_event_tick = tick
                elif current[index] > 0 and next_volume == 0:
                    # Was on, but now turned off
                    evt = midi.NoteOffEvent(
                        tick=(tick - last_event_tick) * step,
                        pitch=index[0]
                    )
                    track.append(evt)
                    last_event_tick = tick

                elif current[index] > 0 and next_volume > 0 and replay[tick][index[0]] > 0:
                    # Handle replay
                    evt_off = midi.NoteOffEvent(
                        tick=(tick- last_event_tick) * step,
                        pitch=index[0]
                    )
                    track.append(evt_off)
                    evt_on = midi.NoteOnEvent(
                        tick=0,
                        velocity=int(volume[tick][index[0]] * MAX_VELOCITY),
                        pitch=index[0]
                    )
                    track.append(evt_on)
                    last_event_tick = tick

        else:
            noop_ticks += 1

        current = data

    tick += 1

    # Turn off all remaining on notes
    for index, vol in np.ndenumerate(current):
        if vol > 0:
            # Was on, but now turned off
            evt = midi.NoteOffEvent(
                tick=(tick - last_event_tick) * step,
                pitch=index[0]
            )
            track.append(evt)
            last_event_tick = tick
            noop_ticks = 0

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=noop_ticks)
    track.append(eot)

    return pattern

def load_midi(fname):
    p = midi.read_midifile(fname)
    cache_path = os.path.join(CACHE_DIR, fname + '.npy')
    try:
        note_seq = np.load(cache_path)
    except Exception as e:
        # Perform caching
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        note_seq = midi_decode(p)
        np.save(cache_path, note_seq)

    assert len(note_seq.shape) == 3, note_seq.shape
    assert note_seq.shape[1] == MIDI_MAX_NOTES, note_seq.shape
    assert note_seq.shape[2] == 3, note_seq.shape
    assert (note_seq >= 0).all()
    assert (note_seq <= 1).all()
    return note_seq

def clamp_midi(sequence):
    """
    Clamps the midi base on the MIN and MAX notes
    """
    return sequence[:, MIN_NOTE:MAX_NOTE, :]


def stagger(data, time_steps):
    dataX, dataY = [], []
    # Buffer training for first event
    data = ([np.zeros_like(data[0])] * time_steps) + list(data)

    # Chop a sequence into measures
    for i in range(0, len(data) - time_steps, NOTES_PER_BAR):
        dataX.append(data[i:i + time_steps])
        dataY.append(data[i + 1:(i + time_steps + 1)])
    return dataX, dataY

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

def compute_beat(beat, notes_in_bar):
    return one_hot(beat % notes_in_bar, notes_in_bar)

def build_or_load(allow_load=True):
    from model import build_models
    models = build_models()
    if allow_load:
        try:
            models[0].load_state_dict(torch.load(MODEL_FILE))
            models[1].style_l = models[0].style_l
            models[1].time_axis = models[0].time_axis
            models[1].dropout1 = models[0].dropout1
            models[1].dropout2 = models[0].dropout2
            models[2].style_l = models[0].style_l
            models[2].naxis = models[0].naxis
            print('Loaded model from file.')
        except:
            print('Unable to load model from file.')
    return models
