import torch

class RepeatVector(torch.nn.Module):
    """
    Input shape: 2D tensor of shape (num_samples, features). Output shape: 3D tensor of shape (num_samples, n, features).
    parameter: 
      n: number of input repeated

    Example:
      # now: input.size() == (batch, features)
      RepeatVector(n)(input)
      # now: input.size() == (batch, n, features)

    """
    def __init__(self, n, **kwargs):
      super(RepeatVector, self).__init__(**kwargs)
      self.n = n
    def forward(self, input):
      input = torch.unsqueeze(input, dim=1)
      input = input.repeat(1, self.n, 1)
      return input