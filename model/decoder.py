import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from typing import Optional, Any, Tuple
from model.attention import MultiHeadAttention
from model.modules import Linear
from model.sublayers import AddNorm, BaseRNN


class Seq2seqDecoder(BaseRNN):
    """
    Converts higher level features (from listener) into output utterances
    by specifying a probability distribution over sequences of characters.

    Args:
        num_classes (int): the number of classfication
        max_length (int): a maximum allowed length for the sequence to be processed
        hidden_dim (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        num_layers (int, optional): number of recurrent layers (default: 1)
        rnn_type (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability (default: 0)
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: inputs, encoder_outputs, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_dim): tensor with containing the outputs of the listener.
          Used for attention mechanism (default is `None`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Returns: decoder_outputs, ret_dict
        - **decoder_outputs** (seq_len, batch, num_classes): list of tensors containing
          the outputs of the decoding function.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_ATTENTION_SCORE* : list of scores
          representing encoder outputs, *KEY_SEQUENCE_SYMBOL* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """
    KEY_ATTENTION_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE_SYMBOL = 'sequence_symbol'

    def __init__(
            self,
            num_classes: int,
            max_decoding_step: int = 120,
            hidden_dim: int = 1024,
            num_heads: int = 4,
            num_layers: int = 2,
            rnn_type: str = 'lstm',
            dropout_p: float = 0.3,
            device: str = 'cuda'
    ) -> None:
        super(Seq2seqDecoder, self).__init__(hidden_dim, hidden_dim, num_layers, rnn_type, dropout_p, False, device)
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.max_decoding_step = max_decoding_step
        self.input_dropout = nn.Dropout(dropout_p)
        self.attention = AddNorm(MultiHeadAttention(hidden_dim, num_heads), hidden_dim)
        self.projection = AddNorm(Linear(hidden_dim, hidden_dim, bias=True), hidden_dim)
        self.generator = Linear(hidden_dim, num_classes, bias=False)

    def forward_step(self, input_var: Tensor, hidden: Optional[Any],
                     encoder_outputs: Tensor) -> Tuple[Tensor, Optional[Any], Tensor]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)

        #embedded = self.embedding(input_var).to(self.device)
        input_var = self.input_dropout(input_var)

        if self.training:
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(input_var, hidden)
        context, attn = self.attention(output, encoder_outputs, encoder_outputs)

        output = self.projection(context.view(-1, self.hidden_dim)).view(batch_size, -1, self.hidden_dim)
        output = self.generator(torch.tanh(output).contiguous().view(-1, self.hidden_dim))

        #step_output = F.log_softmax(output, dim=1)
        step_output = output.view(batch_size, output_lengths, -1).squeeze(1)

        return step_output, hidden, attn

    def forward(
            self,
            inputs: Tensor,
            encoder_outputs: Tensor,
    ) -> Tuple[Tensor, dict]:
        decoder_outputs, ret_dict, hidden = list(), dict(), None

        if not self.training:
            ret_dict[Seq2seqDecoder.KEY_ATTENTION_SCORE] = list()
            ret_dict[Seq2seqDecoder.KEY_SEQUENCE_SYMBOL] = list()

        inputs, batch_size, max_decoding_step = self.validate_args(inputs, encoder_outputs)
        #use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        lengths = np.array([max_decoding_step] * batch_size)


        input_var = inputs.unsqueeze(1)

        for di in range(max_decoding_step):
            step_output, hidden, attn = self.forward_step(input_var, hidden, encoder_outputs)

            decoder_outputs.append(step_output)
            input_var = decoder_outputs[-1].unsqueeze(1)
        ret_dict[Seq2seqDecoder.KEY_LENGTH] = lengths

        return decoder_outputs, ret_dict

    def validate_args(
            self,
            inputs: Optional[Any],
            encoder_outputs: Tensor
    ) -> Tuple[Tensor, int, int]:
        batch_size = encoder_outputs.size(0)
        if inputs is None:  # inference
            inputs = torch.zeros(self.hidden_dim).unsqueeze(0).repeat(batch_size,1)
            max_decoding_step = self.max_decoding_step

            if torch.cuda.is_available():
                inputs = inputs.cuda()
        else:
            max_decoding_step = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_decoding_step
