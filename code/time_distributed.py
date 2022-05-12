'''
PyTorch does not have the implementation of TimeDistributed layer.
The following is the modification of the codes provided from the Pytorch Github issues by the user erogol (ticket number #1927).

Source: https://github.com/pytorch/pytorch/issues/1927
'''
import torch.nn as nn

class TimeDistributed(nn.Module):
    def __init__(self, module, conv1d=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.conv1d = conv1d

    def forward(self, x):
        #print(x.size())
        if len(x.size()) <= 2:
            return self.module(x)
        #print('TimeDistributed input shape: ' + str(x.size()))
        #print(x.type())
        if x.dim() == 4:
            b, t, n, f = x.size(0), x.size(1), x.size(2), x.size(3)
            #x_reshape = x.contiguous().view(x.size()[0], x.size()[-1], -1)
            x_reshape = x.contiguous().view(b, -1, f)
            if self.conv1d: x_reshape = x.contiguous().view(b, f, -1)
            #print('TimeDistributed input shape to the layer: ' + str(x_reshape.size()))
            #print(x_reshape.type())
            y = self.module(x_reshape)
            
            if type(y) is tuple:
                y = y[0]
            #print(y.dim())
            if y.dim() < 4:
                y = y.contiguous().view(b, -1,  y.size(-1))
            else:
                y = y.contiguous().view(b, t, f, y.size(-1))
            #print('TimeDistributed final y shape: ' + str(y.size()))
            return y
        else:
            b, t, f = x.size(0), x.size(1), x.size(2)
            x_reshape = x.contiguous().view(b, -1, f)
            #print('TimeDistributed input shape to the layer: ' + str(x_reshape.size()))
            #print(x_reshape.type())
            y = self.module(x_reshape)
            #print('TimeDistributed y shape: ' + str(y.size()))
            if y.dim() < 4:
                y = y.contiguous().view(b, -1, y.size(-1))
            else:
                y = y.contiguous().view(b, t, -1, y.size(-1))
            #print('TimeDistributed final y shape: ' + str(y.size()))
            return y
