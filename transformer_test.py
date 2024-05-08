import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# Hyperparameters
# Hyperparameters adjusted for M1 Mac capabilities
batch_size = 32  # Reduced batch size to ensure it fits in memory and allows for faster computation
block_size = 128  # Can keep the same if it fits the model’s context needs
max_iters = 500   # Suitable for initial tests
eval_interval = 1  # More frequent evaluations to monitor progress more closely
learning_rate = 3e-4  # Generally a good starting point, adjust based on the obgity served convergence
device = 'mps'  # Using Apple Metal Performance Shaders for training on M1 GPU
n_embd = 256  # Slightly reduced to save memory and compute resources
n_head = 4    # Reduced to decrease complexity and improve speed
n_layer = 4   # Fewer layers to test initial training performance without overloading the system
dropout = 0.1  # Slightly lower to compensate for smaller model and less data per batch
torch.manual_seed(1337)

# Define early stopping parameters
early_stopping_patience = 10  # Number of eval intervals to wait for improvement
target_loss = 0.001  # Target loss threshold for early stopping
best_val_loss = float('inf')
patience_counter = 0

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

kmers = read_kmers('/Users/admin/ProtienBert/train_subset.txt')
stoi, itos = create_vocabulary(kmers)
data_indices = torch.tensor(encode(kmers, stoi), dtype=torch.long).to(device)
train_data, val_data, test_data = train_val_test_split(data_indices)

model = TransformerModel(len(stoi), n_embd, n_head, n_layer, dropout).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

def get_batch(data):
    start_indices = torch.randint(0, len(data) - block_size, (batch_size,))
    sequences = torch.stack([data[start:start + block_size] for start in start_indices])
    targets = torch.stack([data[start + 1:start + block_size + 1] for start in start_indices])
    mask = torch.rand(sequences.shape) < 0.15
    sequences[mask] = stoi['[MASK]']
    return sequences, targets, mask

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
   # Check for early stopping conditions
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset counter
        else:
            patience_counter += 1

        if val_loss <= target_loss:
            print("Early stopping triggered: Target validation loss achieved.")
            break
        elif patience_counter >= early_stopping_patience:
            print("Early stopping triggered: No improvement in validation loss.")
            break

test_loss = evaluate(model, test_data)
print(f"Final Test Loss: {test_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'ProtienBert.pth')
print("Model saved successfully.")
