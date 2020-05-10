import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from utilities import get_pretrained_vectors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LEN = 17


class PolicyNetwork(nn.Module):

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=512, hidden_dim=512, dtype=np.float32, pretrained_embeddings=None):
        super(PolicyNetwork, self).__init__()

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        if pretrained_embeddings is not None:
            vectors = get_pretrained_vectors(pretrained_embeddings)
            self.caption_embedding = nn.Embedding.from_pretrained(vectors)
            for param in self.caption_embedding.parameters():
                param.requires_grad = False
            wordvec_dim = vectors.shape[1]
        else:
            self.caption_embedding = nn.Embedding(vocab_size, wordvec_dim)

        self.cnn2linear = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(wordvec_dim, hidden_dim, batch_first=True)
        self.linear2vocab = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):

        input_captions = self.caption_embedding(captions)
        hidden_init = self.cnn2linear(features)
        cell_init = torch.zeros_like(hidden_init)

        output, _ = self.lstm(input_captions, (hidden_init, cell_init))
        output = self.linear2vocab(output)

        return output


class ValueNetworkRNN(nn.Module):

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=512, hidden_dim=512, dtype=np.float32, pretrained_embeddings=None):
        super(ValueNetworkRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        vocab_size = len(word_to_idx)

        if pretrained_embeddings is not None:
            vectors = get_pretrained_vectors(pretrained_embeddings)
            self.caption_embedding = nn.Embedding.from_pretrained(vectors)
            for param in self.caption_embedding.parameters():
                param.requires_grad = False
            wordvec_dim = vectors.shape[1]
        else:
            self.caption_embedding = nn.Embedding(vocab_size, wordvec_dim)

        self.init_hidden()
        self.lstm = nn.LSTM(wordvec_dim, hidden_dim)

    def init_hidden(self):
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_dim).to(device), torch.zeros(1, 1, self.hidden_dim).to(device))

    def forward(self, captions):

        input_captions = self.caption_embedding(captions)
        output, self.hidden_cell = self.lstm(input_captions.view(len(input_captions), 1, -1), self.hidden_cell)

        return output
    

class ValueNetwork(nn.Module):

    def __init__(self, word_to_idx, pretrained_embeddings):

        super(ValueNetwork, self).__init__()
        self.valrnn = ValueNetworkRNN(word_to_idx)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 1)

    def forward(self, features, captions):

        for t in range(captions.shape[1]):
            value_rnn_output = self.valrnn(captions[:, t])
        value_rnn_output = value_rnn_output.squeeze(0).squeeze(1)
        state = torch.cat((features, value_rnn_output), dim=1)

        output = self.linear1(state)
        output = self.linear2(output)

        return output


class RewardNetworkRNN(nn.Module):

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=512, hidden_dim=512, dtype=np.float32, pretrained_embeddings=None):

        super(RewardNetworkRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        vocab_size = len(word_to_idx)

        if pretrained_embeddings is not None:
            vectors = get_pretrained_vectors(pretrained_embeddings)
            self.caption_embedding = nn.Embedding.from_pretrained(vectors)
            for param in self.caption_embedding.parameters():
                param.requires_grad = False
            wordvec_dim = vectors.shape[1]
        else:
            self.caption_embedding = nn.Embedding(vocab_size, wordvec_dim)

        self.init_hidden()
        self.gru = nn.GRU(wordvec_dim, hidden_dim)

    def init_hidden(self):
        self.hidden_cell = torch.zeros(1, 1, self.hidden_dim).to(device)

    def forward(self, captions):

        input_captions = self.caption_embedding(captions)
        output, self.hidden_cell = self.gru(input_captions.view(len(input_captions), 1, -1), self.hidden_cell)

        return output


class RewardNetwork(nn.Module):

    def __init__(self, word_to_idx, pretrained_embeddings=None):

        super(RewardNetwork, self).__init__()
        self.rewrnn = RewardNetworkRNN(word_to_idx, pretrained_embeddings=pretrained_embeddings)
        self.visual_embed = nn.Linear(512, 512)
        self.semantic_embed = nn.Linear(512, 512)

    def forward(self, features, captions):

        for t in range(captions.shape[1]):
            reward_rnn_output = self.rewrnn(captions[:, t])
        reward_rnn_output = reward_rnn_output.squeeze(0).squeeze(1)
        se = self.semantic_embed(reward_rnn_output)
        ve = self.visual_embed(features)

        return ve, se


class AdvantageActorCriticNetwork(nn.Module):

    def __init__(self, value_network, policy_network):
        super(AdvantageActorCriticNetwork, self).__init__()

        self.value_network = value_network
        self.policy_network = policy_network

    def forward(self, features, captions):
        # Get value from value network
        values = self.value_network(features, captions)
        # Get action probabilities from policy network
        probs = self.policy_network(features.unsqueeze(0), captions)[:, -1:, :]
        return values, probs

