import torch
from torch import nn
from torch.nn import init

############
## LAYERS ##
############

def initialize_weights(m):
  if isinstance(m, nn.Linear):
    init.xavier_uniform(m.weight.data)

#######################
#  AUDIO ONLY MODELS  #
#######################

class SimpleAudioRNN(nn.Module):
  def __init__(self, config):
    super(SimpleAudioRNN, self).__init__()
    self.config = config
    self.rnn = nn.LSTM(68, config.hidden_size, batch_first = True)
    self.decoder =  nn.Linear(config.hidden_size, config.num_classes)

  def forward(self, input):
    seq_output, hidden = self.rnn(input)
    hidden_state, cell_state = hidden
    decoded = self.decoder(hidden_state.squeeze(0))
    #output = pad_packed_sequence(output, batch_first = True)
    return decoded 

