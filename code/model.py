'''
Reimplementation of model.py (https://github.com/calclavia/DeepJ/blob/icsc/model.py) with PyTorch.

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
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanSquaredError
from time_distributed import *
from RepeatVector import *
#from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import TimeDistributed
#from keras import losses
from utility import *
from constants import *

def primary_loss(y_true, y_pred):
    played = y_true[:, :, :,0]
    bce_note = nn.BCELoss()(y_pred[:, :, :, 0], y_true[:, :, :, 0])
    bce_replay = nn.BCELoss()(torch.mul(played, y_pred[:, :, :, 1]) + torch.mul(1 - played, y_true[:, :, :, 1]), y_true[:, :, :, 1])
    mse = nn.MSELoss()(torch.mul(played, y_pred[:, :, :, 2]) + torch.mul(1 - played, y_true[:, :, :, 2]), y_true[:, :, :, 2])
    return bce_note + bce_replay + mse

def pitch_pos_in_f(time_steps):
    """
    Returns a constant containing pitch position of each note
    """
    def f(x):
        note_ranges = torch.arange(0, NUM_NOTES, dtype=torch.float32) / NUM_NOTES
        repeated_ranges = note_ranges.repeat(x.size()[0] * time_steps)
        repeated_ranges = (repeated_ranges.view([x.size(0), time_steps, NUM_NOTES, 1])).to(torch.device("cuda"))
        #repeated_ranges = (repeated_ranges.view([x.size(0), time_steps, NUM_NOTES, 1]))
        return repeated_ranges
    return f

def pitch_class_in_f(time_steps):
    """
    Returns a constant containing pitch class of each note
    """
    def f(x):
        pitch_class_matrix = np.array([np.array(one_hot(n % OCTAVE, OCTAVE)) for n in range(NUM_NOTES)])
        pitch_class_matrix = torch.as_tensor(pitch_class_matrix, device=torch.device("cuda"), dtype=torch.float32)
        #pitch_class_matrix = torch.as_tensor(pitch_class_matrix, dtype=torch.float32)
       	pitch_class_matrix = pitch_class_matrix.view([1, 1, NUM_NOTES, OCTAVE])
        pitch_class_matrix = (pitch_class_matrix.repeat(x.size()[0], time_steps, 1, 1))
        return pitch_class_matrix
    return f

def pitch_bins_f(time_steps):
    def f(x):
        bins = torch.sum(torch.stack([x[:, :, i::OCTAVE, 0] for i in range(OCTAVE)]), dim=3)
        bins = bins.repeat(NUM_OCTAVES, 1, 1)
        bins = bins.view([x.size()[0], time_steps, NUM_NOTES, 1])
        return bins
    return f

class MainModel(nn.Module):
    def __init__(self, time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
        super().__init__()
        self.dropout1 = nn.Dropout(p=input_dropout)
        self.dropout2 = nn.Dropout(p=input_dropout)
        self.dropout3 = nn.Dropout(p=input_dropout)
        self.dropout4 = nn.Dropout(p=dropout)
        self.dropout5 = nn.Dropout(p=dropout)
        self.dropout6 = nn.Dropout(p=dropout)
        self.dropout7 = nn.Dropout(p=dropout)
        self.dropout8 = nn.Dropout(p=dropout)

        kernel_size = 2 * OCTAVE
        self.style_l = nn.Linear(NUM_STYLES, STYLE_UNITS, device=torch.device('cuda')).float()

        self.conv1d = nn.Conv1d(in_channels=3, out_channels=OCTAVE_UNITS, stride=1, kernel_size=kernel_size, padding='same', device=torch.device('cuda')).float()
        self.time_distributed = TimeDistributed(self.conv1d, conv1d=True)

        self.repeat_vector = RepeatVector(NUM_NOTES)
        self.time_distributed2 = TimeDistributed(self.repeat_vector)

        self.repeat_vector2 = RepeatVector(NUM_NOTES)
        self.time_distributed3 = TimeDistributed(self.repeat_vector2)

        self.dense1 = nn.Linear(STYLE_UNITS, 94, device=torch.device('cuda')).float()
        self.dense2 = nn.Linear(STYLE_UNITS, 256, device=torch.device('cuda')).float()

        self.lstm1 = nn.LSTM(input_size=94, hidden_size=TIME_AXIS_UNITS, batch_first=True).to(torch.device("cuda")).float()
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=TIME_AXIS_UNITS, batch_first=True).to(torch.device("cuda")).float()
        self.time_distributed4 = TimeDistributed(self.lstm1)
        self.time_distributed5 = TimeDistributed(self.lstm2)

        self.dense3 = nn.Linear(STYLE_UNITS, 259, device=torch.device("cuda")).float()
        self.dense4 = nn.Linear(STYLE_UNITS, 128, device=torch.device("cuda")).float()

        self.lstm3 = nn.LSTM(input_size=259, hidden_size=NOTE_AXIS_UNITS, batch_first=True).to(torch.device("cuda")).float()
        self.lstm4 = nn.LSTM(input_size=128, hidden_size=NOTE_AXIS_UNITS, batch_first=True).to(torch.device("cuda")).float()

        self.time_distributed6 = TimeDistributed(self.lstm3)
        self.time_distributed7 = TimeDistributed(self.lstm4)

        self.repeat_vector3 = RepeatVector(NUM_NOTES)
        self.time_distributed8 = TimeDistributed(self.repeat_vector3)

        self.dense5 = nn.Linear(128, 2, device=torch.device("cuda")).float()
        self.dense6 = nn.Linear(128, 1, device=torch.device("cuda")).float()


    def forward(self, notes_in, chosen_in, beat_in, style_in):
        notes = self.dropout1(notes_in)
        beats = self.dropout2(beat_in)
        chosen = self.dropout3(chosen_in)
        style = self.style_l(style_in)

        time_steps = int(notes.size()[1])

        note_octave = self.time_distributed(notes)
        note_octave = torch.tanh(note_octave)
        note_octave = self.dropout4(note_octave)

        pitch_pos = pitch_pos_in_f(time_steps)(notes)
        pitch_class = pitch_class_in_f(time_steps)(notes)
        pitch_bins = pitch_bins_f(time_steps)(notes)

        beat_repeat = self.time_distributed2(beats)
        note_octave = note_octave.view(pitch_pos.size()[0], pitch_pos.size()[1], pitch_pos.size()[2], -1)
        beat_repeat = beat_repeat.view(pitch_pos.size()[0], pitch_pos.size()[1], pitch_pos.size()[2], -1)

        note_features = torch.cat([pitch_pos, pitch_class, pitch_bins, note_octave, beat_repeat], -1)
        note_features = torch.permute(note_features, (0, 2, 1, 3))
        # Apply LSTMs
        for l in range(TIME_AXIS_LAYERS):
            # Integrate style
            style_proj = None

            if l == 0:
                style_proj = self.dense1(style).float()
            else:
                style_proj = self.dense2(style).float()

            style_proj = self.time_distributed3(style_proj)
            style_proj = torch.tanh(style_proj)
            style_proj = self.dropout5(style_proj)
            style_proj = torch.permute(style_proj, (0, 2, 1, 3))

            note_features = torch.add(note_features, style_proj)

            if l == 0:
                note_features = self.time_distributed4(note_features)
            else:
                note_features = self.time_distributed5(note_features)

            note_features = self.dropout6(note_features)
            note_features = note_features.view(style_proj.size(0), style_proj.size(1), style_proj.size(2), -1)

        time_out = torch.permute(note_features, (0, 2, 1, 3))



        time_steps = int(time_out.size(1))
        # Shift target one note to the left.
        pad = lambda time_out: F.pad(time_out[:, :, :-1, :], (0, 0, 1, 0, 0, 0, 0, 0))

        shift_chosen = pad(chosen)
        shift_chosen = shift_chosen.view(time_out.size(0), time_steps, NUM_NOTES, -1)

        # [batch, time, notes, features + 1]
        time_out = torch.cat((time_out, shift_chosen), 3)
        for l in range(NOTE_AXIS_LAYERS):
            # Integrate style
            style_proj = None
            if l == 0:
                style_proj = self.dense3(style)
            else:
                style_proj = self.dense4(style)

            style_proj = self.time_distributed8(style_proj)
            style_proj = torch.tanh(style_proj)
            style_proj = self.dropout7(style_proj)
            time_out = torch.add(time_out, style_proj)

            if l == 0:
                time_out = self.time_distributed6(time_out)
            else:
                time_out = self.time_distributed7(time_out)

            time_out = self.dropout8(time_out)
            time_out = time_out.view(style_proj.size(0), style_proj.size(1), style_proj.size(2), -1)

        note_densed = torch.sigmoid(self.dense5(time_out))
        volume_densed = self.dense6(time_out)
        return torch.cat((note_densed, volume_densed), -1)


class TimeModel(nn.Module):
    def __init__(self, input_dropout=0.2):
        super().__init__()
        self.dropout1 = nn.Dropout(p=input_dropout)
        self.dropout2 = nn.Dropout(p=input_dropout)

        self.style_l = nn.Linear(NUM_STYLES, STYLE_UNITS, device=torch.device('cuda')).float()

        kernel_size = 2 * OCTAVE

        self.conv1d = nn.Conv1d(in_channels=3, out_channels=OCTAVE_UNITS, stride=1, kernel_size=kernel_size, padding='same', device=torch.device('cuda')).float()
        self.time_distributed = TimeDistributed(self.conv1d, conv1d=True)

        self.repeat_vector = RepeatVector(NUM_NOTES)
        self.time_distributed2 = TimeDistributed(self.repeat_vector)

        self.repeat_vector2 = RepeatVector(NUM_NOTES)
        self.time_distributed3 = TimeDistributed(self.repeat_vector2)

        self.dense1 = nn.Linear(STYLE_UNITS, 94, device=torch.device('cuda')).float()
        self.dense2 = nn.Linear(STYLE_UNITS, 256, device=torch.device('cuda')).float()

        self.lstm1 = nn.LSTM(input_size=94, hidden_size=TIME_AXIS_UNITS, batch_first=True).to(torch.device("cuda")).float()
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=TIME_AXIS_UNITS, batch_first=True).to(torch.device("cuda")).float()

        self.time_distributed4 = TimeDistributed(self.lstm1)
        self.time_distributed5 = TimeDistributed(self.lstm2)

    def forward(self, notes_in, beat_in, style_in):
        notes = self.dropout1(notes_in)
        beats = self.dropout2(beat_in)
        style = self.style_l(style_in)

        time_steps = int(notes.size()[1])

        note_octave = self.time_distributed(notes)
        note_octave = torch.tanh(note_octave)
        note_octave = self.dropout4(note_octave)

        pitch_pos = pitch_pos_in_f(time_steps)(notes)
        pitch_class = pitch_class_in_f(time_steps)(notes)
        pitch_bins = pitch_bins_f(time_steps)(notes)

        beat_repeat = self.time_distributed2(beat)
        note_octave = note_octave.view(pitch_pos.size()[0], pitch_pos.size()[1], pitch_pos.size()[2], -1)
        beat_repeat = beat_repeat.view(pitch_pos.size()[0], pitch_pos.size()[1], pitch_pos.size()[2], -1)

        note_features = torch.cat([pitch_pos, pitch_class, pitch_bins, note_octave, beat_repeat], -1)
        note_features = torch.permute(note_features, (0, 2, 1, 3))

        # Apply LSTMs
        for l in range(TIME_AXIS_LAYERS):
            # Integrate style
            style_proj = None

            if l == 0:
                style_proj = self.dense1(style).float()
            else:
                style_proj = self.dense2(style).float()

            style_proj = self.time_distributed3(style_proj)
            style_proj = torch.tanh(style_proj)
            style_proj = self.dropout5(style_proj)
            style_proj = torch.permute(style_proj, (0, 2, 1, 3))
            note_features = torch.add(note_features, style_proj)

            if l == 0:
                note_features = self.time_distributed4(note_features)
            else:
                note_features = self.time_distributed5(note_features)

            note_features = self.dropout6(note_features)
            note_features = note_features.view(style_proj.size(0), style_proj.size(1), style_proj.size(2), -1)

        time_out = torch.permute(note_features, (0, 2, 1, 3))
        return time_out

class NoteModel(nn.Module):
    def __init__(self, input_dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=input_dropout)
        self.style_l = nn.Linear(NUM_STYLES, STYLE_UNITS, device=torch.device('cuda')).float()

        self.dense3 = nn.Linear(STYLE_UNITS, 259, device=torch.device("cuda")).float()
        self.dense4 = nn.Linear(STYLE_UNITS, 128, device=torch.device("cuda")).float()

        self.lstm3 = nn.LSTM(input_size=259, hidden_size=NOTE_AXIS_UNITS, batch_first=True).to(torch.device("cuda")).float()
        self.lstm4 = nn.LSTM(input_size=128, hidden_size=NOTE_AXIS_UNITS, batch_first=True).to(torch.device("cuda")).float()

        self.time_distributed6 = TimeDistributed(self.lstm3)
        self.time_distributed7 = TimeDistributed(self.lstm4)

        self.repeat_vector3 = RepeatVector(NUM_NOTES)
        self.time_distributed8 = TimeDistributed(self.repeat_vector3)

        self.dense5 = nn.Linear(128, 2, device=torch.device("cuda")).float()
        self.dense6 = nn.Linear(128, 1, device=torch.device("cuda")).float()

    def forward(self, note_features, chosen_gen_in, style_gen_in):
        chosen = self.dropout(chosen_gen_in)
        style = self.style_l(style_gen_in)
        note_gen_out = self.naxis(note_features, chosen_gen, style_gen)

        time_steps = int(note_features.size(1))
        # Shift target one note to the left.
        pad = lambda note_features: F.pad(note_features[:, :, :-1, :], (0, 0, 1, 0, 0, 0, 0, 0))

        shift_chosen = pad(chosen)
        shift_chosen = shift_chosen.view(note_features.size(0), time_steps, NUM_NOTES, -1)

        # [batch, time, notes, features + 1]
        note_features = torch.cat((note_features, shift_chosen), 3)
        for l in range(NOTE_AXIS_LAYERS):
            # Integrate style
            style_proj = None
            if l == 0:
                style_proj = self.dense3(style)
            else:
                style_proj = self.dense4(style)

            style_proj = self.time_distributed8(style_proj)
            style_proj = torch.tanh(style_proj)
            style_proj = self.dropout7(style_proj)
            note_features = torch.add(note_features, style_proj)

            if l == 0:
                note_features = self.time_distributed6(note_features)
            else:
                note_features = self.time_distributed7(note_features)

            note_features = self.dropout8(note_features)
            note_features = note_features.view(style_proj.size(0), style_proj.size(1), style_proj.size(2), -1)

        note_densed = torch.sigmoid(self.dense5(note_features))
        volume_densed = self.dense6(note_features)
        return torch.cat((note_densed, volume_densed), -1)

def build_models(time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
    model = MainModel().to(torch.device('cuda'))
    time_model = TimeModel().to(torch.device('cuda'))
    note_model = NoteModel().to(torch.device('cuda'))
    return model.float(), time_model.float(), note_model.float()
