'''
Reimplementation of generate.py (https://github.com/calclavia/DeepJ/blob/icsc/generate.py) with PyTorch.

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
import torch
import numpy as np
from collections import deque
import midi
import argparse

from constants import *
from utility import *
#from dataset import *
from tqdm import tqdm
from utility import midi_encode

class MusicGeneration:
    """
    Represents a music generation
    """
    def __init__(self, style, default_temp=1):
        self.notes_memory = deque([np.zeros((NUM_NOTES, NOTE_UNITS)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.beat_memory = deque([np.zeros(NOTES_PER_BAR) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.style_memory = deque([style for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)

        # The next note being built
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        self.silent_time = NOTES_PER_BAR

        # The outputs
        self.results = []
        # The temperature
        self.default_temp = default_temp
        self.temperature = default_temp

    def build_time_inputs(self):
        return (
            np.array(self.notes_memory),
            np.array(self.beat_memory),
            np.array(self.style_memory)
        )

    def build_note_inputs(self, note_features):
        # Timesteps = 1 (No temporal dimension)
        return (
            np.array(note_features),
            np.array([self.next_note]),
            np.array(list(self.style_memory)[-1:])
        )

    def choose(self, prob, n):
        vol = prob[n, -1]
        prob = apply_temperature(prob[n, :-1], self.temperature)

        # Flip notes randomly
        if np.random.random() <= prob[0]:
            self.next_note[n, 0] = 1
            # Apply volume
            self.next_note[n, 2] = vol
            # Flip articulation
            if np.random.random() <= prob[1]:
                self.next_note[n, 1] = 1

    def end_time(self, t):
        """
        Finish generation for this time step.
        """
        # Increase temperature while silent.
        if np.count_nonzero(self.next_note) == 0:
            self.silent_time += 1
            if self.silent_time >= NOTES_PER_BAR:
                self.temperature += 0.1
        else:
            self.silent_time = 0
            self.temperature = self.default_temp

        self.notes_memory.append(self.next_note)
        # Consistent with dataset representation
        self.beat_memory.append(compute_beat(t, NOTES_PER_BAR))
        self.results.append(self.next_note)
        # Reset next note
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        return self.results[-1]

def apply_temperature(prob, temperature):
    """
    Applies temperature to a sigmoid vector.
    """
    # Apply temperature
    if temperature != 1:
        # Inverse sigmoid
        x = -np.log(1 / prob - 1)
        # Apply temperature to sigmoid function
        prob = 1 / (1 + np.exp(-x / temperature))
    return prob

def process_inputs(ins):
    ins = list(zip(*ins))
    ins = [np.array(i) for i in ins]
    return ins

def get_predicted_value(model, samples):
    predicted_values = list()
    model.eval()
    for index in range(samples.size(0)):
        current_sample = torch.tensor(samples[index], dtype=torch.float32, device=torch.device('cuda'))
        for value in model[current_sample]:
            predicted_values.append(value)
    return predicted_values

def generate(models, num_bars, styles):
    print('Generating with styles:', styles)

    _, time_model, note_model = models
    generations = [MusicGeneration(style) for style in styles]

    for t in tqdm(range(NOTES_PER_BAR * num_bars)):
        # Produce note-invariant features
        ins = process_inputs([g.build_time_inputs() for g in generations])
        # Pick only the last time step
        note_features = get_predicted_value(time_model, ins)
        note_features = np.array(note_features)[:, -1:, :]

        # Generate each note conditioned on previous
        for n in range(NUM_NOTES):
            ins = process_inputs([g.build_note_inputs(note_features[i, :, :, :]) for i, g in enumerate(generations)])
            predictions = np.array(note_model.predict(ins))

            for i, g in enumerate(generations):
                # Remove the temporal dimension
                g.choose(predictions[i][-1], n)

        # Move one time step
        yield [g.end_time(t) for g in generations]

def write_file(name, results):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    results = zip(*list(results))

    for i, result in enumerate(results):
        fpath = os.path.join(SAMPLES_DIR, name + '_' + str(i) + '.mid')
        print('Writing file', fpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        mf = midi_encode(unclamp_midi(result))
        midi.write_midifile(fpath, mf)

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--bars', default=32, type=int, help='Number of bars to generate')
    parser.add_argument('--styles', default=None, type=int, nargs='+', help='Styles to mix together')
    args = parser.parse_args()

    models = build_or_load()

    styles = [compute_genre(i) for i in range(len(genre))]

    if args.styles:
        # Custom style
        styles = [np.mean([one_hot(i, NUM_STYLES) for i in args.styles], axis=0)]

    write_file('output', generate(models, args.bars, styles))

if __name__ == '__main__':
    main()
