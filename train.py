import os
import numpy as np
import torch
from torch import nn
import pickle

# Load data
def load_data(data_folder='txt_files/'):
    all_text = ''
    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(data_folder, filename), 'r', encoding='utf-8') as f:
                all_text += f.read() + '\n'
    return all_text

# Create character mappings
def create_mappings(all_text):
    chars = tuple(set(all_text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    return chars, int2char, char2int

# Encode data
def encode_text(all_text, char2int):
    return np.array([char2int[ch] for ch in all_text])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, tokens, int2char, char2int, n_hidden=512, n_layers=2, drop_prob=0.5):
        super(DeckTalkRNN, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.chars = tokens  # chars parameter from mappings
        self.int2char = int2char
        self.char2int = char2int

        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        x = x.to(device).float()
        hidden = tuple([h.to(device) for h in hidden])
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.reshape(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
    

def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot

def train(model, data, epochs=10, batch_size=64, seq_length=100, lr=0.001, clip=5, print_every=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for e in range(epochs):
        hidden = model.init_hidden(batch_size)
        hidden = tuple([each.to(device) for each in hidden])

        for batch_i, (x, y) in enumerate(get_batches(data, batch_size, seq_length)):
            x = one_hot_encode(x, len(chars))
            inputs = torch.from_numpy(x).to(device)
            targets = torch.from_numpy(y).to(device)

            hidden = tuple([each.data for each in hidden])
            hidden = tuple([each.to(device) for each in hidden])

            model.zero_grad()
            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(batch_size * seq_length).long())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if batch_i % print_every == 0:
                print(f"Epoch: {e+1}/{epochs}... Step: {batch_i}... Loss: {loss.item():.4f}")

if __name__ == '__main__':
    all_text = load_data('txt_files/')
    chars, int2char, char2int = create_mappings(all_text)
    encoded_text = encode_text(all_text, char2int)

    model = DeckTalkRNN(chars, int2char, char2int, n_hidden=512, n_layers=2)
    model.to(device)

    train(model, encoded_text, epochs=20, batch_size=64, seq_length=70, lr=0.001)

    model_filename = 'models/decktalk_rnn_70.pth'
    torch.save(model.state_dict(), model_filename)
    print(f'Model saved to {model_filename}')

    with open('models/char_mappings_70.pkl', 'wb') as f:
        pickle.dump({'chars': chars, 'int2char': int2char, 'char2int': char2int}, f)
