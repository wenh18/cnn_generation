import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN

def Improved_SemHash(select_weight):
    # select_weight:[batch_size,length,kernel_number]
    binary_selection = torch.lt(torch.zeros_like(select_weight), select_weight).float()
    gradient_selection = torch.max(torch.zeros_like(select_weight), torch.min(torch.ones_like(select_weight), (
                1.2 * torch.sigmoid(select_weight) - 0.1)))
    d = binary_selection + gradient_selection - gradient_selection.detach()
    return d

class Seq2seq(nn.Module):
    """
    Standard sequence-to-sequence architecture with configurable encoder and decoder.

    Args:
        encoder (torch.nn.Module): encoder of seq2seq
        decoder (torch.nn.Module): decoder of seq2seq

    Inputs: inputs, input_lengths, targets, teacher_forcing_ratio
        - **inputs** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (torch.Tensor): tensor of sequences, whose contains length of inputs.
        - **targets** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0.90)

    Returns: decoder_outputs, ret_dict
        - **decoder_outputs** (seq_len, batch, num_classes): list of tensors containing
          the outputs of the decoding function.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_ATTENTION_SCORE* : list of scores
          representing encoder outputs, *KEY_SEQUENCE_SYMBOL* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: Tensor, input_lengths: Optional[Tensor] = None,) -> Tuple[Tensor, dict]:
        encoder_outputs, hidden = self.encoder(inputs, input_lengths)
        # import pdb;pdb.set_trace()
        result = self.decoder(inputs=None,
                      encoder_hidden=hidden,
                      encoder_outputs=encoder_outputs)

        decision = torch.stack(result[0]).permute(1, 0, 2)
        final_decision = Improved_SemHash(decision)
        # import pdb;pdb.set_trace()
        # comparor = torch.ones_like(decision) * 0.5
        # final_decision = torch.gt(decision, comparor).float()

        return final_decision

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def set_speller(self, decoder: nn.Module):
        self.decoder = decoder



def get_seq2seq_model(description_size=100, max_len=48, hidden_size=128, bidirectional=True, output_size=2048):

    '''
    description_size: the length of an description
    max_len: number of kernels to be determined in one backbone
    output_size: maximum channel size in one kernel
    hidden_size: encoder output dimension
    '''
    
    encoder = EncoderRNN(description_size, max_len, hidden_size,
                          bidirectional=bidirectional, variable_lengths=False)
    decoder = DecoderRNN(output_size, max_len, hidden_size * 2 if bidirectional else hidden_size,
                          dropout_p=0.2, use_attention=True, bidirectional=bidirectional)
                          # eos_id=tgt.eos_id, sos_id=tgt.sos_id)
    seq2seq = Seq2seq(encoder, decoder)

    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)
    return seq2seq

if __name__ == '__main__':
    # x=torch.ones(32, 10, 100).long().cuda()
    x=torch.ones(32, 10, 100).cuda()
    model=get_seq2seq_model().cuda()
    y=model(x)
    # print(y[0].shape)
    import pdb;pdb.set_trace()

