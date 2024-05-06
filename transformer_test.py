import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# Hyperparameters
batch_size = 64
block_size = 128
max_iters = 500
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
torch.manual_seed(1337)

# Reading and encoding functions
def read_kmers(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def create_vocabulary(kmers):
    unique_kmers = set(kmers)
    unique_kmers.add('[MASK]')  # Make sure the [MASK] token is included
    stoi = {kmer: i for i, kmer in enumerate(unique_kmers)}
    itos = {i: kmer for i, kmer in enumerate(unique_kmers)}
    return stoi, itos

def encode(kmers, stoi):
    return [stoi.get(kmer, stoi['[MASK]']) for kmer in kmers]

def train_val_test_split(data, train_frac=0.7, val_frac=0.15):
    train_end = int(len(data) * train_frac)
    val_end = train_end + int(len(data) * val_frac)
    return data[:train_end], data[train_end:val_end], data[val_end:]

# Transformer Model Definition
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embeddings = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dropout=dropout, dim_feedforward=4 * n_embd) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, mask=None):
        x = self.embedding(idx) + self.positional_embeddings[:, :idx.size(1), :]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        if mask is not None:
            loss = F.cross_entropy(logits[mask], idx[mask])
            return logits, loss
        return logits

# Load data and prepare for training
kmers = read_kmers('train_subset.txt')
stoi, itos = create_vocabulary(kmers)
data_indices = torch.tensor(encode(kmers, stoi), dtype=torch.long).to(device)
train_data, val_data, test_data = train_val_test_split(data_indices)

# Initialize model and optimizer
model = TransformerModel(len(stoi), n_embd, n_head, n_layer, dropout).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Batch preparation function
def get_batch(data):
    start_indices = torch.randint(0, len(data) - block_size, (batch_size,))
    sequences = torch.stack([data[start:start + block_size] for start in start_indices])
    targets = torch.stack([data[start + 1:start + block_size + 1] for start in start_indices])
    mask = torch.rand(sequences.shape) < 0.15
    sequences[mask] = stoi['[MASK]']
    return sequences, targets, mask

# Training and evaluation loop
def evaluate(model, data):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(0, len(data), batch_size):
            sequences, targets, mask = get_batch(data)
            _, loss = model(sequences.to(device), mask.to(device))
            total_loss += loss.item()
    return total_loss / (len(data) / batch_size)

for i in range(max_iters):
    model.train()
    sequences, targets, mask = get_batch(train_data)
    optimizer.zero_grad()
    _, loss = model(sequences.to(device), mask.to(device))
    loss.backward()
    optimizer.step()

    if i % eval_interval == 0:
        val_loss = evaluate(model, val_data)
        print(f"Iteration {i}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")

# Final evaluation on the test dataset
test_loss = evaluate(model, test_data)
print(f"Final Test Loss: {test_loss:.4f}")
