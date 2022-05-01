'''
PyTorch does not have the implementation of TimeDistributed layer.
The following is the implementation provided from the Pytorch Github issues by the user erogol (ticket number #1927).

Source: https://github.com/pytorch/pytorch/issues/1927
'''
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        t, n = x.size(0), x.size(1) 
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(t * n, x.size(2))
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(t, n, y.size()[1])
        return y