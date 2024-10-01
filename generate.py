import torch
from torch import nn
import os
import numpy as np
import pickle
from train import DeckTalkRNN  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load character mappings
with open('char_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)
    chars = mappings['chars']
    int2char = mappings['int2char']
    char2int = mappings['char2int']

# Recreate the model instance
n_hidden = 512
n_layers = 2
model = DeckTalkRNN(chars=chars, n_hidden=n_hidden, n_layers=n_layers)
model.load_state_dict(torch.load('decktalk_rnn.pth', map_location=device))
model.to(device)
model.eval()

def one_hot_encode(arr, n_labels):
    # Your one_hot_encode function here
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot

def predict(model, char, hidden=None, temperature=1.0):
    x = np.array([[model.char2int[char]]])
    x = one_hot_encode(x, len(model.chars))
    inputs = torch.from_numpy(x).to(device)
    
    hidden = tuple([each.data for each in hidden])
    hidden = tuple([each.to(device) for each in hidden])
    
    out, hidden = model(inputs, hidden)
    prob = nn.functional.softmax(out / temperature, dim=1).data
    prob = prob.cpu()
    
    char_ind = np.random.choice(len(model.chars), p=prob.numpy().squeeze())
    return model.int2char[char_ind], hidden

def sample(model, size, prime='[:', temperature=1.0):
    model.eval()
    chars = [ch for ch in prime]
    hidden = model.init_hidden(1)
    hidden = tuple([each.to(device) for each in hidden])

    for ch in prime:
        char, hidden = predict(model, ch, hidden, temperature)

    chars.append(char)

    for _ in range(size):
        char, hidden = predict(model, chars[-1], hidden, temperature)
        chars.append(char)

    return ''.join(chars)

generated_text = sample(model, size=1000, prime='[:', temperature=0.8)
print(generated_text)
