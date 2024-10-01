import os
import numpy as np
import torch
from torch import nn


# Load data
data_folder = 'txt_files/'
all_text = ''

for filename in os.listdir(data_folder):
    if filename.endswith('.txt'):
        with open(os.path.join(data_folder, filename), 'r', encoding='utf-8') as f:
            all_text += f.read() + '\n'

# Create charicter mapping 
chars = tuple(set(all_text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

# encode data 
encoded_text = np.array([char2int[ch] for ch in all_text])

# Hayperpraramiters
sequence_length = 100 
batch_size = 64

def get_batches(arr, batch_size, seq_length):
    batch_size_total = batch_size * seq_length
    n_batches = len(arr) // batch_size_total

    arr = arr[:n_batches * batch_size_total]
    arr = arr.reshape((batch_size, -1))

    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n:n+seq_length]
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


class DeckTalkRNN(nn.Module):
    def __init__(self, tokens, n_hidden=512, n_layers=2, drop_prob=0.5, lr=0.001):
        super(DeckTalkRNN, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.chars = tokens
        self.int2char = int2char
        self.char2int = char2int

        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        x = x.float()
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.reshape(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_())


model = DeckTalkRNN(chars, n_hidden=512, n_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train(model, data, epochs=10, batch_size=64, seq_length=100, lr=0.001, clip=5, print_every=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for e in range(epochs):
        hidden = model.init_hidden(batch_size)

        for batch_i, (x, y) in enumerate(get_batches(data, batch_size, seq_length)):
            x = one_hot_encode(x, len(chars))
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            hidden = tuple([each.data for each in hidden])

            model.zero_grad()
            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(batch_size * seq_length).long())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if batch_i % print_every == 0:
                print(f"Epoch: {e+1}/{epochs}... Step: {batch_i}... Loss: {loss.item()}")

def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


def predict(model, char, hidden=None, temperature=1.0):
    x = np.array([[model.char2int[char]]])
    x = one_hot_encode(x, len(model.chars))
    inputs = torch.from_numpy(x)

    hidden = tuple([each.data for each in hidden])
    out, hidden = model(inputs, hidden)

    prob = nn.functional.softmax(out / temperature, dim=1).data
    prob = prob.cpu()

    char_ind = np.random.choice(len(model.chars), p=prob.numpy().squeeze())
    return model.int2char[char_ind], hidden

def sample(model, size, prime='[:', temperature=1.0):
    model.eval()
    chars = [ch for ch in prime]
    hidden = model.init_hidden(1)

    for ch in prime:
        char, hidden = predict(model, ch, hidden, temperature)

    chars.append(char)

    for _ in range(size):
        char, hidden = predict(model, chars[-1], hidden, temperature)
        chars.append(char)

    return ''.join(chars)


train(model, encoded_text, epochs=20, batch_size=64, seq_length=100, lr=0.001)
# print(sample(model, 1000, prime='[:', temperature=0.8))