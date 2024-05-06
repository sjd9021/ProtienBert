import numpy as np
import random
from sklearn.model_selection import train_test_split

# Define parameters
k = 3  # Length of each k-mer
step = 1  # Step size for overlapping

# Function to generate k-mers from a sequence
def generate_kmers(sequence, k, step):
    return [sequence[i:i+k] for i in range(0, len(sequence) - k + 1, step)]

# Function to read protein sequences from a file
def read_protein_sequences(filepath):
    with open(filepath, 'r') as file:
        sequences = [line.strip() for line in file if line.strip()]
    print("Number of protein sequences read:", len(sequences))
    return sequences

# Function to write k-mers to a text file
def write_kmers_to_txt(kmers, filename):
    with open(filename, 'w') as file:
        for kmer in kmers:
            file.write(f"{kmer}\n")
    print(f"k-mers have been written to {filename}")

# Function to mask k-mers
def mask_kmers(kmers, mask_token="[MASK]", mask_probability=0.15):
    masked_kmers = []
    labels = []
    for kmer in kmers:
        if random.random() < mask_probability:
            masked_kmers.append(mask_token)
            labels.append(kmer)  # the original k-mer is the label for training
        else:
            masked_kmers.append(kmer)
            labels.append(None)  # None means no prediction needed here
    return masked_kmers, labels

# Path to the file containing protein sequences
file_path = '/Users/admin/ProtienBert/parsed_sequences.txt'
protein_sequences = read_protein_sequences(file_path)

# Generate k-mers for all sequences
all_kmers = [km for seq in protein_sequences for km in generate_kmers(seq, k, step)]
print("Example k-mers:", all_kmers[:10])

# Masking k-mers
masked_kmers, labels = mask_kmers(all_kmers)
print("Masked k-mers sample (first 10):", masked_kmers[:10])

# Write original and masked k-mers to text files
original_kmer_file_path = '/Users/admin/ProtienBert/kmers3.txt'
masked_kmer_file_path = '/Users/admin/ProtienBert/kmers_masked3.txt'
write_kmers_to_txt(all_kmers, original_kmer_file_path)
write_kmers_to_txt(masked_kmers, masked_kmer_file_path)

# Optional: Tokenize k-mers into characters if required by the model
# tokenized_kmers = [[char for char in kmer] for kmer in masked_kmers]
# print("Tokenized k-mers sample (first 10):", tokenized_kmers[:10])
