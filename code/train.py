'''
Reimplementation of train.py (https://github.com/calclavia/DeepJ/blob/icsc/train.py) with PyTorch.

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
#from keras.callbacks import ModelCheckpoint, LambdaCallback
#from keras.callbacks import EarlyStopping, TensorBoard
import argparse
import midi
import os
import numpy as np

from constants import *
from preprocess import *
from generate import *
#from midi_util import midi_encode
from model import *
from preprocess import *

def main():
    print('torch cuda available: ' + str(torch.cuda.is_available()))
    models = build_or_load()
    train(models)

def train(models):
    print('Loading data ...')
    train_labels = None
    note_data = None
    note_target = None
    beat_data = None
    style_data = None

    if os.path.exists('../data/note_data.npy'):
        print('Preprocessed data found ...')
        note_data = np.load('../data/note_data.npy', 'r')
        note_target = np.load('../data/note_target.npy', 'r')
        beat_data = np.load('../data/beat_data.npy', 'r')
        style_data = np.load('../data/style_data.npy', 'r')
        train_labels = note_target
    else:
        print('Calling the preprocessor ...')
        train_data, train_labels = get_preprocessed_data(styles, BATCH_SIZE, SEQ_LEN)
        note_data = train_data[0]
        note_target = train_data[1]
        beat_data = train_data[2]
        style_data = train_data[3]

    print('Training ...')

    optimizer = torch.optim.NAdam(models[0].parameters())
    loss_function = primary_loss

    for i in range(1000):
        print('epoch #' + str(i + 1) + '/' + str(1000))
    	index = 0
    	while (index + BATCH_SIZE) < len(train_labels):
            print('Batch #' + str(i + 1))
            current_note_data = torch.as_tensor(note_data[index:index + BATCH_SIZE], dtype=torch.float64, device=torch.device('cuda'))
            current_note_target = torch.as_tensor(note_target[index:index + BATCH_SIZE], dtype=torch.float64, device=torch.device('cuda'))
            current_beat_data = torch.as_tensor(beat_data[index:index + BATCH_SIZE], dtype=torch.float64, device=torch.device('cuda'))
            current_style_data = torch.as_tensor(style_data[index:index + BATCH_SIZE], dtype=torch.float64, device=torch.device('cuda'))
            current_labels = torch.as_tensor(train_labels[index:index + BATCH_SIZE], dtype=torch.float64, device=torch.device('cuda'))
            predicted_labels = models[0](current_note_data, current_note_target, current_beat_data, current_style_data)
            loss = loss_function(predicted_labels, current_labels)
            print('Loss: ' + str(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            index += BATCH_SIZE

if __name__ == '__main__':
    main()