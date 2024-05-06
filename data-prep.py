import numpy as np
from sklearn.model_selection import train_test_split

# Define parameters
k = 5  # Length of each k-mer
step = 1  # Step size for overlapping

# Function to generate k-mers from a sequence
def generate_kmers(sequence, k, step):
    return [sequence[i:i+k] for i in range(0, len(sequence) - k + 1, step)]

# Function to read protein sequences from a file
def read_protein_sequences(filepath):
    with open(filepath, 'r') as file:
        return [line.strip() for line in file if line.strip()]

# Function to write k-mers to a text file
def write_kmers_to_txt(kmers, filename):
    with open(filename, 'w') as file:
        for kmer in kmers:
            file.write(f"{kmer}\n")

# Path to the file containing protein sequences
file_path = '/Users/admin/ProtienBert/parsed_sequences.txt'
protein_sequences = read_protein_sequences(file_path)
print("Number of protein sequences read:", len(protein_sequences))  # Print how many sequences were read

# Generate k-mers for all sequences
all_kmers = [km for seq in protein_sequences for km in generate_kmers(seq, k, step)]
print("Example k-mers:", all_kmers[:10])  # Print first 10 k-mers to verify

# Write k-mers to a text file
kmer_file_path = '/Users/admin/ProtienBert/kmers5.txt'
write_kmers_to_txt(all_kmers, kmer_file_path)
print(f"k-mers have been written to {kmer_file_path}")

# Tokenization (example with amino acids as tokens)
stoi = {char: idx for idx, char in enumerate(sorted(set(''.join(protein_sequences))))}
tokenized_kmers = [[stoi[char] for char in km] for km in all_kmers]
print("Tokenized k-mers example:", tokenized_kmers[:10])  # Print first 10 tokenized k-mers

# Split into training and validation sets
train_kmers, val_kmers = train_test_split(tokenized_kmers, test_size=0.1, random_state=42)
print("Number of training k-mers:", len(train_kmers))  # Check the size of the training dataset
print("Number of validation k-mers:", len(val_kmers))  # Check the size of the validation dataset

# Example: preparing batches (simple batching example, without padding)
batch_size = 10
train_batches = [train_kmers[i:i + batch_size] for i in range(0, len(train_kmers), batch_size)]
print("First training batch:", train_batches[0])  # Print the first batch to check

# Additional simple test to ensure the batch sizes are as expected
assert all(len(batch) == batch_size for batch in train_batches[:-1]), "All batches except possibly the last should have the correct batch size"
print("All training batches (except possibly the last) are of the correct batch size.")
