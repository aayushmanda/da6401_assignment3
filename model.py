import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, cell_type="LSTM", dropout=0.0):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        self.embedding = nn.Embedding(input_size, hidden_size)

        if cell_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif cell_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size, device):
        if self.cell_type == "LSTM":
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, cell_type="LSTM", dropout=0.0):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        self.embedding = nn.Embedding(output_size, hidden_size)

        if cell_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif cell_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded, hidden)
        output = self.softmax(self.out(output))
        return output, hidden

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_hidden = self.encoder.init_hidden(batch_size, self.device)
        _, encoder_hidden = self.encoder(src, encoder_hidden)

        decoder_input = src[:, 0:1]
        decoder_hidden = encoder_hidden

        for t in range(1, trg_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t:t+1, :] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.argmax(2)
            decoder_input = trg[:, t:t+1] if teacher_force else top1

        return outputs


