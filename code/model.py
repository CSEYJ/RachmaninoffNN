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
    #print('true label: ' + str(y_true.size()))
    #print('predicted label: ' + str(y_pred.size()))
    # 3 separate loss calculations based on if note is played or not
    played = y_true[:, :, :,0]
    #print('1')
    #print(y_pred[:,:,:,0])
    #print(y_pred[:,:,:,0])
    #print(y_true[:,:,:,0])
    bce_note = nn.BCELoss()(y_pred[:, :, :, 0], y_true[:, :, :, 0])
    #print(bce_note)
    #print('2')
    #print(torch.mul(played, y_pred[:, :, :, 1]))
    #print(played)
    #print('pred:')
    #print(y_pred[:,:,:,1])
    #print('true:')
    #print(y_true.size())
    #print(y_true[:,:,:,1])
    #print('loss')
    bce_replay = nn.BCELoss()(torch.mul(played, y_pred[:, :, :, 1]) + torch.mul(1 - played, y_true[:, :, :, 1]), y_true[:, :, :, 1])
    #print(bce_replay)
    #print('3')
    mse = nn.MSELoss()(torch.mul(played, y_pred[:, :, :, 2]) + torch.mul(1 - played, y_true[:, :, :, 2]), y_true[:, :, :, 2])
    #print(mse)
    #print(torch.mul(played, y_pred[:, :, :, 2]))
    #print('4')
    #print(bce_note + bce_replay + mse)
    return bce_note + bce_replay + mse

def pitch_pos_in_f(time_steps):
    """
    Returns a constant containing pitch position of each note
    """
    def f(x):
        note_ranges = torch.arange(0, NUM_NOTES, dtype=torch.float32) / NUM_NOTES
        repeated_ranges = note_ranges.repeat(x.size()[0] * time_steps)
        repeated_ranges = (repeated_ranges.view([x.size(0), time_steps, NUM_NOTES, 1])).to(torch.device("cuda"))
        #print('pitch pos shape: ' + str(repeated_ranges.size()))
        return repeated_ranges
    return f

def pitch_class_in_f(time_steps):
    """
    Returns a constant containing pitch class of each note
    """
    def f(x):
        pitch_class_matrix = np.array([np.array(one_hot(n % OCTAVE, OCTAVE)) for n in range(NUM_NOTES)])
       	#print(pitch_class_matrix)
        pitch_class_matrix = torch.as_tensor(pitch_class_matrix, device=torch.device("cuda"), dtype=torch.float32)
       	pitch_class_matrix = pitch_class_matrix.view([1, 1, NUM_NOTES, OCTAVE])
        #print(pitch_class_matrix.size())
        pitch_class_matrix = (pitch_class_matrix.repeat(x.size()[0], time_steps, 1, 1))
        #print('pitch class shape: ' + str(pitch_class_matrix.size()))
        return pitch_class_matrix
    return f

def pitch_bins_f(time_steps):
	def f(x):
                #print([x[:,:,i::OCTAVE,0] for i in range(OCTAVE)])
                bins = torch.sum(torch.stack([x[:, :, i::OCTAVE, 0] for i in range(OCTAVE)]), dim=3)
                #print(bins.size())
                bins = bins.repeat(NUM_OCTAVES, 1, 1)
                bins = bins.view([x.size()[0], time_steps, NUM_NOTES, 1])
                #print('bin shape: ' + str(bins.size()))
                return bins
	return f

def time_axis(dropout):
    def f(notes, beat, style):
        time_steps = int(notes.size()[1])
        kernel_size = 2 * OCTAVE
        timeDistributed = TimeDistributed(nn.Conv1d(in_channels=notes.size(-1), out_channels=OCTAVE_UNITS, stride=1, kernel_size=kernel_size, padding='same', device=torch.device('cuda')).float(), conv1d=True)
        note_octave = timeDistributed(notes)
        note_octave = torch.tanh(note_octave)
        note_octave = nn.Dropout(p=dropout)(note_octave)
        # Create features for every single note.
        pitch_pos = pitch_pos_in_f(time_steps)(notes)
        pitch_class = pitch_class_in_f(time_steps)(notes)
        pitch_bins = pitch_bins_f(time_steps)(notes)
        beat_repeat = TimeDistributed(RepeatVector(NUM_NOTES))(beat)
        #print(beat_repeat.type())
        note_octave = note_octave.view(pitch_pos.size()[0], pitch_pos.size()[1], pitch_pos.size()[2], -1)
        beat_repeat = beat_repeat.view(pitch_pos.size()[0], pitch_pos.size()[1], pitch_pos.size()[2], -1)
        #print('pitch pos: ' + str(pitch_pos.size()) + '\npitch_class: ' + str(pitch_class.size()) + '\npitch bin: ' + str(pitch_bins.size()) + '\note octave: ' + str(note_octave.size()) + '\nbeat repeat: ' + str(beat_repeat.size()))
        note_features = torch.cat([pitch_pos, pitch_class, pitch_bins, note_octave, beat_repeat], -1)
        x = note_features
        #print('permuting')
        #print('x shape: ' + str(x.size()))
        #print(x.type())
        x = torch.permute(x, (0, 2, 1, 3))
        #print('permuted x shape: ' + str(x.size()))

        # Apply LSTMs
        for l in range(TIME_AXIS_LAYERS):
            # Integrate style
            #print('calling linear on style: ' + str(style.size()))
            style_proj = nn.Linear(STYLE_UNITS, int(x.size(-1)), device=torch.device('cuda'))(style).float()
            #print('calling time distributed: ' + str(style_proj.size()))
            style_proj = TimeDistributed(RepeatVector(NUM_NOTES))(style_proj)
            #print(style_proj.type())
            #print('calling tanh: ' + str(style_proj.size()))
            style_proj = torch.tanh(style_proj)
            style_proj = nn.Dropout(p=dropout)(style_proj)
            style_proj = torch.permute(style_proj, (0, 2, 1, 3))
            #x = x.view(style_proj.size(0), style_proj.size(1), style_proj.size(2), style_proj.size(3))
            x = torch.add(x, style_proj)
            x = TimeDistributed(nn.LSTM(input_size=int(x.size(-1)), hidden_size=TIME_AXIS_UNITS, batch_first=True).to(torch.device("cuda")).float())(x)
            x = nn.Dropout(p=dropout)(x)
            x = x.view(style_proj.size(0), style_proj.size(1), style_proj.size(2), -1)

        return torch.permute(x, (0, 2, 1, 3))
    return f

def note_axis(dropout):
    dense_layer_cache = {}
    lstm_layer_cache = {}
    
    def f(x, chosen, style):
        time_steps = int(x.size(1))

        # Shift target one note to the left.
        pad = lambda x: F.pad(x[:, :, :-1, :], (0, 0, 1, 0, 0, 0, 0, 0))
        shift_chosen = pad(chosen)

        shift_chosen = shift_chosen.view(x.size(0), time_steps, NUM_NOTES, -1)
        # [batch, time, notes, features + 1]
        x = torch.cat((x, shift_chosen), 3)

        for l in range(NOTE_AXIS_LAYERS):
            # Integrate style
            if l not in dense_layer_cache:
            	dense_layer_cache[l] = nn.Linear(STYLE_UNITS, int(x.size(-1)), device=torch.device("cuda")).float()

            style_proj = dense_layer_cache[l](style)
            style_proj = TimeDistributed(RepeatVector(NUM_NOTES))(style_proj)
            style_proj = torch.tanh(style_proj)
            style_proj = nn.Dropout(p=dropout)(style_proj)
            x = torch.add(x, style_proj)

            if l not in lstm_layer_cache:
                lstm_layer_cache[l] = nn.LSTM(input_size=int(x.size(-1)), hidden_size=NOTE_AXIS_UNITS, batch_first=True).to(torch.device("cuda"))

            x = TimeDistributed(lstm_layer_cache[l].float())(x)
            x = nn.Dropout(p=dropout)(x)
            x = x.view(style_proj.size(0), style_proj.size(1), style_proj.size(2), -1)

        note_dense = nn.Linear(x.size(-1), 2, device=torch.device("cuda")).float()
        #note_dense = nn.Linear(x.size(-1), 2).float()
        note_densed = torch.sigmoid(note_dense(x))
        volume_dense = nn.Linear(x.size(-1), 1, device=torch.device("cuda")).float()
        #volumne_dense = nn.Linear(x.size(-1), 1).float()
        volume_densed = volume_dense(x)
        return torch.cat((note_densed, volume_densed), -1)
    return f

class MainModel(nn.Module):
    def __init__(self, time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
        super().__init__()
        self.dropout1 = nn.Dropout(p=input_dropout)
        self.dropout2 = nn.Dropout(p=input_dropout)
        self.dropout3 = nn.Dropout(p=input_dropout)
        self.style_l = nn.Linear(NUM_STYLES, STYLE_UNITS, device=torch.device('cuda')).float()
        #self.style_l = nn.Linear(NUM_STYLES, STYLE_UNITS).float()
        self.time_axis = time_axis(dropout)
        self.naxis = note_axis(dropout)

    def forward(self, notes_in, chosen_in, beat_in, style_in):
        #print(notes_in.size())
        notes = self.dropout1(notes_in)
        #print(notes.size())
        beats = self.dropout2(beat_in)
        chosen = self.dropout3(chosen_in)
        style = self.style_l(style_in)
        time_out = self.time_axis(notes, beats, style)
        notes_out = self.naxis(time_out, chosen, style)
        return notes_out

class TimeModel(nn.Module):
	def __init__(self, time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
		super().__init__()
		self.dropout1 = nn.Dropout(p=input_dropout)
		self.dropout2 = nn.Dropout(p=input_dropout)
		self.style_l = nn.Linear(NUM_STYLES, STYLE_UNITS).double()
		self.time_axis = time_axis(dropout)

	def forward(self, notes_in, beat_in, style_in):
		notes = self.dropout1(notes_in)
		beats = self.dropout2(beat_in)
		style = self.style_l(style_in)
		time_out = self.time_axis(notes, beats, style)
		return time_out

class NoteModel(nn.Module):
	def __init__(self, time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
		super().__init__()
		self.dropout = nn.Dropout(p=input_dropout)
		self.style_l = nn.Linear(NUM_STYLES, STYLE_UNITS).double()
		self.naxis = note_axis(dropout)

	def forward(self, note_features, chosen_gen_in, style_gen_in):
		chosen_gen = self.dropout(chosen_gen_in)
		style_gen = self.style_l(style_gen_in)
		note_gen_out = self.naxis(note_features, chosen_gen, style_gen)
		return note_gen_out

def build_models(time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
    model = MainModel().to(torch.device('cuda'))
    time_model = TimeModel().to(torch.device('cuda'))
    note_model = NoteModel().to(torch.device('cuda'))
    #model = MainModel()
    #time_model = TimeModel()
    #note_model = NoteModel()
    return model.float(), time_model.float(), note_model.float()
