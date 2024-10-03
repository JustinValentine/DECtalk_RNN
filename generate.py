import torch
from torch import nn
import numpy as np
import pickle
from train import DeckTalkRNN 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('models/char_mappings_70.pkl', 'rb') as f:
    mappings = pickle.load(f)
chars = mappings['chars']
int2char = mappings['int2char']
char2int = mappings['char2int']

n_hidden = 512
n_layers = 2
model = DeckTalkRNN(tokens=chars, int2char=int2char, char2int=char2int, n_hidden=n_hidden, n_layers=n_layers) # Instantiate the model
model.load_state_dict(torch.load('models/decktalk_rnn_70.pth', map_location=device, weights_only=True)) # Load the trained model's weights

model.to(device)
model.eval()

def one_hot_encode(arr, n_labels):
    arr = torch.tensor(arr, dtype=torch.long, device=device)
    one_hot = torch.zeros((*arr.shape, n_labels), device=device)
    one_hot.scatter_(-1, arr.unsqueeze(-1), 1.0)
    return one_hot

# Predict the next character given a character and hidden state
def predict(model, char, hidden=None, temperature=1.0):
    with torch.no_grad():
        x = np.array([[model.char2int[char]]])
        x = one_hot_encode(x, len(model.chars))
        hidden = tuple([each.data.to(device) for each in hidden])

        # Forward pass through the model with the input and hidden state
        out, hidden = model(x, hidden)
        out = out / temperature # Adjust the output values based on the temperature parameter

        prob = nn.functional.softmax(out, dim=1) # Tensor containing the probabilities of each character being the next one

        # Sample the next character index from the probability distribution
        char_ind = torch.multinomial(prob, num_samples=1).item()
        return model.int2char[char_ind], hidden # Predicted character and the updated hidden state

def sample(model, size, start_seq='[:', temperature=1.0):
    model.eval()
    chars = [ch for ch in start_seq]

    hidden = model.init_hidden(1)# Batch size is 1 for text generation
    hidden = tuple([each.to(device) for each in hidden])

    for ch in start_seq:
        char, hidden = predict(model, ch, hidden, temperature)

    # Append last predicted character from the prime sequence to the list
    chars.append(char)

    for _ in range(size):
        # Use the last generated character to predict the next one
        char, hidden = predict(model, chars[-1], hidden, temperature)
        chars.append(char)
        
    return ''.join(chars)

# Generate text
generated_text = sample(model, size=1000, start_seq='[:', temperature=0.8)
print(generated_text)