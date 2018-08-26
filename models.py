import torch
from torch import nn
from torch.nn import init

############
## LAYERS ##
############

def initialize_weights(m):
  if isinstance(m, nn.Linear):
    init.xavier_uniform_(m.weight.data)

#######################
#  AUDIO ONLY MODELS  #
#######################

class Flatten(nn.Module):
  def forward(self, x):
      N = x.size(0) # read in N, C, H, W
      return x.contiguous().view(N, -1) # "flatten" the C * H * W values into a single vector per image

class AudioRNN(nn.Module):
  def __init__(self, config, feature_size = 68):
    super(AudioRNN, self).__init__()
    # Do not change model, copy and paste into new class.
    self.config = config
    self.rnn = nn.LSTM(feature_size, config.hidden_size, batch_first = True)
    self.flat_dim = config.max_length * config.hidden_size
    self.decoder =  nn.Sequential(
        Flatten(),
        nn.Linear(self.flat_dim, config.hidden_size*2),
        nn.BatchNorm1d(config.hidden_size*2),
        nn.ReLU(),
        nn.Linear(config.hidden_size*2, config.num_classes)
      )

  def forward(self, input):
    seq_output, hidden = self.rnn(input)
    #hidden_state, cell_state = hidden
    decoded = self.decoder(seq_output)
    #output = pad_packed_sequence(output, batch_first = True)
    return decoded 

