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
import argparse
import midi
import os
import numpy as np
from constants import *
from preprocess import *
from generate import *
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
        train_labels = np.load('../data/note_target.npy', 'r')
    else:
        print('Calling the preprocessor ...')
        train_data, train_labels = get_preprocessed_data(styles, BATCH_SIZE, SEQ_LEN)
        note_data = train_data[0]
        note_target = train_data[1]
        beat_data = train_data[2]
        style_data = train_data[3]

    nepoch = 0
    index = 0
    if os.path.exists(EPOCH_FILE):
        temp = open(EPOCH_FILE, "r").readline()
        if len(temp) > 0: nepoch = int(temp)

    if os.path.exists(INDEX_FILE):
        temp = open(INDEX_FILE, "r").readline()
        if len(temp) > 0: index = int(temp)

    if not os.path.exists('./out'): 
        os.makedirs('./out')

    
    print('Training ...')

    optimizer = torch.optim.NAdam(models[0].parameters(), lr=0.001)
    loss_function = primary_loss
    nepoch = 0
    for i in range(nepoch, 1000):
        epoch_file = open(EPOCH_FILE, 'w')
        epoch_file.write(str(i))
        epoch_file.close()

        print('epoch #' + str(i + 1) + '/' + str(1000))
        while (index + BATCH_SIZE) < len(train_labels):
            print('Batch #' + str(int(index/BATCH_SIZE + 1)) + '/' + str(int(len(train_labels)/BATCH_SIZE)))
            current_note_data = torch.tensor(note_data[index:index + BATCH_SIZE], dtype=torch.float32, device=torch.device('cuda'))
            current_note_target = torch.tensor(note_target[index:index + BATCH_SIZE], dtype=torch.float32, device=torch.device('cuda'))
            current_beat_data = torch.tensor(beat_data[index:index + BATCH_SIZE], dtype=torch.float32, device=torch.device('cuda'))
            current_style_data = torch.tensor(style_data[index:index + BATCH_SIZE], dtype=torch.float32, device=torch.device('cuda'))
            current_labels = torch.tensor(train_labels[index:index + BATCH_SIZE], dtype=torch.float32, device=torch.device('cuda'))
            optimizer.zero_grad()
            predicted_labels = models[0](current_note_data, current_note_target, current_beat_data, current_style_data)
            loss = primary_loss(current_labels, predicted_labels)/3
            print('loss: ' + str(loss.item()))
            loss.backward()
            optimizer.step()
            print('Saving the weights ...')
            torch.save(models[0].state_dict(), MODEL_FILE)
            index += BATCH_SIZE

            index_file = open(INDEX_FILE, 'w')
            index_file.write(str(index))
            index_file.close()

        index = 0

    print('Finished the training ...')
    print('Saving the weights ...')
    torch.save(models[0].state_dict(), MODEL_FILE)


if __name__ == '__main__':
    main()
